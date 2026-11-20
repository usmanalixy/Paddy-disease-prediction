package com.usmanalixy.paddydiseaseprediction;

import android.app.Activity;
import android.content.ActivityNotFoundException;
import android.content.ClipData;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.ColorSpace;
import android.graphics.ImageDecoder;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.core.content.FileProvider;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowCompat;
import androidx.core.view.WindowInsetsCompat;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.IOException;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class MainActivity extends Activity {
    private static final String TAG = "PaddyPrediction";
    private static final int PICK_PHOTO = 1;
    private static final int CAPTURE_PHOTO = 2;
    private static final int MAX_PHOTO_EDGE = 2048;
    private enum PhotoKind { PENDING, LEAF, OTHER }

    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private PaddyClassifier classifier; // Created, used, and closed on worker.
    private JSONObject metadata;
    private Uri selectedPhoto;
    private Bitmap selectedBitmap;
    private PhotoKind photoKind = PhotoKind.PENDING;
    private File selectedCapture;
    private File pendingCapture;
    private boolean waitingForPhoto;
    private boolean busy;

    private Button camera;
    private Button picker;
    private ImageView preview;
    private View placeholder;
    private TextView status;
    private View photoCheck;
    private View results;
    private TextView resultTitle;
    private TextView resultDetail;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        WindowCompat.setDecorFitsSystemWindows(getWindow(), false);
        setContentView(R.layout.activity_main);
        View screen = findViewById(R.id.screen);
        ViewCompat.setOnApplyWindowInsetsListener(screen, (view, insets) -> {
            Insets safe = insets.getInsets(
                    WindowInsetsCompat.Type.systemBars() | WindowInsetsCompat.Type.displayCutout());
            view.setPadding(safe.left, safe.top, safe.right, safe.bottom);
            return insets;
        });
        ViewCompat.requestApplyInsets(screen);

        camera = findViewById(R.id.camera);
        picker = findViewById(R.id.picker);
        preview = findViewById(R.id.preview);
        findViewById(R.id.photo_frame).setClipToOutline(true);
        placeholder = findViewById(R.id.placeholder);
        status = findViewById(R.id.status);
        photoCheck = findViewById(R.id.photo_check);
        results = findViewById(R.id.results);
        resultTitle = findViewById(R.id.result_title);
        resultDetail = findViewById(R.id.result_detail);
        camera.setOnClickListener(view -> takePhoto());
        picker.setOnClickListener(view -> choosePhoto());
        findViewById(R.id.leaf_photo).setOnClickListener(view -> {
            photoKind = PhotoKind.LEAF;
            analyzeConfirmedLeaf();
        });
        findViewById(R.id.other_photo).setOnClickListener(view -> {
            photoKind = PhotoKind.OTHER;
            showResult(R.string.closeup_needed, getString(R.string.closeup_detail));
        });

        try {
            metadata = PaddyClassifier.readMetadata(getAssets());
            JSONObject evaluation = metadata.getJSONObject("health_screen").getJSONObject("test");
            TextView metrics = findViewById(R.id.evaluation);
            int normalCount = evaluation.getInt("normal_images");
            metrics.setText(getResources().getQuantityString(R.plurals.evaluation_summary, normalCount,
                    evaluation.getInt("normal_false_alarms"), normalCount));
        } catch (IOException | JSONException error) {
            Log.e(TAG, "Cannot load model metadata", error);
            showStatus(R.string.error_model, true);
        }

        if (savedInstanceState != null) {
            pendingCapture = savedCapture(savedInstanceState.getString("pending_capture"));
            selectedCapture = savedCapture(savedInstanceState.getString("selected_capture"));
            waitingForPhoto = savedInstanceState.getBoolean("waiting_for_photo");
            photoKind = PhotoKind.valueOf(savedInstanceState.getString("photo_kind", PhotoKind.PENDING.name()));
            String photo = savedInstanceState.getString("selected_photo");
            if (photo != null) {
                selectedPhoto = Uri.parse(photo);
                if (!waitingForPhoto) loadPhoto(selectedPhoto);
            }
        }
    }

    @Override
    protected void onSaveInstanceState(Bundle state) {
        super.onSaveInstanceState(state);
        if (selectedPhoto != null) state.putString("selected_photo", selectedPhoto.toString());
        if (selectedCapture != null) state.putString("selected_capture", selectedCapture.getName());
        if (pendingCapture != null) state.putString("pending_capture", pendingCapture.getName());
        state.putBoolean("waiting_for_photo", waitingForPhoto);
        state.putString("photo_kind", photoKind.name());
    }

    @Override
    protected void onDestroy() {
        worker.execute(() -> {
            if (classifier != null) classifier.close();
        });
        worker.shutdown();
        if (isFinishing()) {
            discardCapture(selectedCapture);
            discardCapture(pendingCapture);
        }
        super.onDestroy();
    }

    private File cameraDirectory() {
        return new File(getCacheDir(), "camera");
    }

    private File savedCapture(String name) {
        return name == null ? null : new File(cameraDirectory(), name);
    }

    private Uri cameraUri(File photo) {
        return FileProvider.getUriForFile(this, getPackageName() + ".photos", photo);
    }

    private void takePhoto() {
        try {
            File directory = cameraDirectory();
            if (!directory.isDirectory() && !directory.mkdirs()) {
                throw new IOException("Cannot create the camera cache.");
            }
            pendingCapture = File.createTempFile("paddy_", ".jpg", directory);
            Uri output = cameraUri(pendingCapture);
            Intent capture = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            capture.putExtra(MediaStore.EXTRA_OUTPUT, output);
            capture.setClipData(ClipData.newRawUri("paddy photo", output));
            capture.addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION | Intent.FLAG_GRANT_READ_URI_PERMISSION);
            startActivityForResult(capture, CAPTURE_PHOTO);
            waitingForPhoto = true;
        } catch (IOException | ActivityNotFoundException | SecurityException error) {
            discardCapture(pendingCapture);
            pendingCapture = null;
            showStatus(R.string.error_camera, true);
        }
    }

    private void choosePhoto() {
        try {
            if (Build.VERSION.SDK_INT >= 33) {
                try {
                    startActivityForResult(new Intent(MediaStore.ACTION_PICK_IMAGES)
                            .setType("image/*"), PICK_PHOTO);
                    waitingForPhoto = true;
                    return;
                } catch (ActivityNotFoundException ignored) {
                    // Some devices ship without the system photo picker.
                }
            }
            Intent document = new Intent(Intent.ACTION_OPEN_DOCUMENT);
            document.setType("image/*");
            document.addCategory(Intent.CATEGORY_OPENABLE);
            startActivityForResult(document, PICK_PHOTO);
            waitingForPhoto = true;
        } catch (ActivityNotFoundException | SecurityException error) {
            showStatus(R.string.error_picker, true);
        }
    }

    @Override
    protected void onActivityResult(int request, int result, Intent data) {
        super.onActivityResult(request, result, data);
        if (request != CAPTURE_PHOTO && request != PICK_PHOTO) return;
        waitingForPhoto = false;
        if (request == CAPTURE_PHOTO && pendingCapture != null) {
            File capture = pendingCapture;
            pendingCapture = null;
            Uri uri = cameraUri(capture);
            revokeUriPermission(uri, Intent.FLAG_GRANT_WRITE_URI_PERMISSION | Intent.FLAG_GRANT_READ_URI_PERMISSION);
            if (result == RESULT_OK && capture.length() > 0) {
                replacePhoto(uri, capture);
                return;
            } else {
                discardCapture(capture);
                if (result == RESULT_OK) showStatus(R.string.error_photo, true);
            }
        } else if (request == PICK_PHOTO && result == RESULT_OK && data != null && data.getData() != null) {
            replacePhoto(data.getData(), null);
            return;
        }
        if (selectedPhoto != null && preview.getVisibility() != View.VISIBLE) {
            loadPhoto(selectedPhoto);
        }
    }

    private void replacePhoto(Uri uri, File capture) {
        discardCapture(selectedCapture);
        selectedCapture = capture;
        photoKind = PhotoKind.PENDING;
        loadPhoto(uri);
    }

    private void discardCapture(File capture) {
        if (capture != null && capture.exists() && !capture.delete()) {
            Log.w(TAG, "Could not clear temporary camera photo");
        }
    }

    private void loadPhoto(Uri photo) {
        if (busy) return;
        selectedPhoto = photo;
        selectedBitmap = null;
        setBusy(true);
        results.setVisibility(View.GONE);
        photoCheck.setVisibility(View.GONE);
        preview.setImageDrawable(null);
        preview.setVisibility(View.GONE);
        placeholder.setVisibility(View.VISIBLE);
        showStatus(R.string.loading_photo, false);

        worker.execute(() -> {
            Bitmap bitmap;
            try {
                bitmap = ImageDecoder.decodeBitmap(ImageDecoder.createSource(getContentResolver(), photo),
                        (decoder, info, source) -> {
                            decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE);
                            decoder.setTargetColorSpace(ColorSpace.get(ColorSpace.Named.SRGB));
                            int edge = Math.max(info.getSize().getWidth(), info.getSize().getHeight());
                            if (edge > MAX_PHOTO_EDGE) {
                                decoder.setTargetSampleSize((edge - 1) / MAX_PHOTO_EDGE + 1);
                            }
                        });
            } catch (IOException | RuntimeException error) {
                runOnUiThread(() -> showPhotoError());
                return;
            }
            runOnUiThread(() -> previewPhoto(bitmap));
        });
    }

    private void previewPhoto(Bitmap bitmap) {
        if (isDestroyed() || isFinishing()) return;
        selectedBitmap = bitmap;
        preview.setImageBitmap(bitmap);
        preview.setVisibility(View.VISIBLE);
        placeholder.setVisibility(View.GONE);
        status.setVisibility(View.GONE);
        setBusy(false);
        switch (photoKind) {
            case PENDING -> photoCheck.setVisibility(View.VISIBLE);
            case LEAF -> analyzeConfirmedLeaf();
            case OTHER -> showResult(R.string.closeup_needed, getString(R.string.closeup_detail));
        }
    }

    private void analyzeConfirmedLeaf() {
        if (busy || selectedBitmap == null || photoKind != PhotoKind.LEAF) return;
        Bitmap bitmap = selectedBitmap;
        setBusy(true);
        photoCheck.setVisibility(View.GONE);
        showStatus(R.string.analyzing, false);
        worker.execute(() -> {
            try {
                if (metadata == null) throw new IOException("Model metadata is unavailable.");
                if (classifier == null) classifier = new PaddyClassifier(getAssets(), metadata);
                PaddyClassifier.Prediction prediction = classifier.predict(bitmap);
                runOnUiThread(() -> showPrediction(prediction));
            } catch (IOException | JSONException | RuntimeException | LinkageError error) {
                Log.e(TAG, "Classifier failed", error);
                runOnUiThread(() -> {
                    if (isDestroyed() || isFinishing()) return;
                    setBusy(false);
                    showStatus(R.string.error_model, true);
                });
            }
        });
    }

    private void showPhotoError() {
        if (isDestroyed() || isFinishing()) return;
        setBusy(false);
        selectedPhoto = null;
        discardCapture(selectedCapture);
        selectedCapture = null;
        showStatus(R.string.error_photo, true);
    }

    private void showPrediction(PaddyClassifier.Prediction prediction) {
        switch (prediction.outcome()) {
            case NORMAL -> showResult(R.string.result_normal, getString(R.string.result_normal_detail));
            case UNCERTAIN -> showResult(R.string.result_uncertain, getString(R.string.result_uncertain_detail));
            case POSSIBLE_DISEASE -> showResult(R.string.result_disease,
                    getString(R.string.result_disease_detail, displayLabel(prediction.diseaseLabel())));
        }
    }

    private void showResult(int title, String detail) {
        if (isDestroyed() || isFinishing()) return;
        resultTitle.setText(title);
        resultDetail.setText(detail);
        photoCheck.setVisibility(View.GONE);
        results.setVisibility(View.VISIBLE);
        status.setVisibility(View.GONE);
        setBusy(false);
    }

    private void setBusy(boolean value) {
        busy = value;
        camera.setEnabled(!value);
        picker.setEnabled(!value);
    }

    private void showStatus(int message, boolean error) {
        status.setText(message);
        status.setTextColor(getColor(error ? R.color.error : R.color.muted));
        status.setVisibility(View.VISIBLE);
    }

    private static String displayLabel(String key) {
        String label = key.replace('_', ' ');
        return label.isEmpty() ? label : label.substring(0, 1).toUpperCase(Locale.ROOT) + label.substring(1);
    }
}
