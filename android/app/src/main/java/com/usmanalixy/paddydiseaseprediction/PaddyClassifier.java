package com.usmanalixy.paddydiseaseprediction;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

final class PaddyClassifier implements AutoCloseable {
    record Prediction(HealthDecision.Outcome outcome, String diseaseLabel) {}

    private final Interpreter model;
    private final ByteBuffer input;
    private final String[] labels;
    private final HealthDecision healthDecision;
    private final int width;
    private final int height;

    static JSONObject readMetadata(AssetManager assets) throws IOException, JSONException {
        try (InputStream stream = assets.open("paddy_model.json")) {
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            byte[] buffer = new byte[4096];
            int length;
            while ((length = stream.read(buffer)) != -1) {
                bytes.write(buffer, 0, length);
            }
            return new JSONObject(new String(bytes.toByteArray(), StandardCharsets.UTF_8));
        }
    }

    PaddyClassifier(AssetManager assets, JSONObject metadata) throws IOException, JSONException {
        JSONArray classes = metadata.getJSONArray("class_names");
        labels = new String[classes.length()];
        for (int i = 0; i < labels.length; i++) {
            labels[i] = classes.getString(i);
        }
        JSONObject screen = metadata.getJSONObject("health_screen");
        JSONObject policy = screen.getJSONObject("policy");
        healthDecision = new HealthDecision(
                Arrays.asList(labels).indexOf(screen.getString("normal_class")),
                (float) policy.getDouble("normal_below"),
                (float) policy.getDouble("disease_above"),
                (float) policy.getDouble("min_disease_class_score"));
        JSONArray imageSize = metadata.getJSONArray("image_size");
        height = imageSize.getInt(0);
        width = imageSize.getInt(1);
        input = ByteBuffer.allocateDirect(height * width * 3 * Float.BYTES)
                .order(ByteOrder.nativeOrder());

        ByteBuffer weights;
        try (AssetFileDescriptor file = assets.openFd("paddy_model.tflite");
             FileChannel channel = file.createInputStream().getChannel()) {
            weights = channel.map(FileChannel.MapMode.READ_ONLY, file.getStartOffset(), file.getLength());
        }
        model = new Interpreter(weights, new Interpreter.Options().setNumThreads(2));
        try {
            Tensor modelInput = model.getInputTensor(0);
            Tensor modelOutput = model.getOutputTensor(0);
            if (modelInput.dataType() != DataType.FLOAT32 || modelOutput.dataType() != DataType.FLOAT32
                    || !Arrays.equals(modelInput.shape(), new int[] {1, height, width, 3})
                    || !Arrays.equals(modelOutput.shape(), new int[] {1, labels.length + 1})) {
                throw new IOException("Model tensors do not match the bundled metadata.");
            }
        } catch (IOException | RuntimeException error) {
            model.close();
            throw error;
        }
    }

    Prediction predict(Bitmap photo) {
        int sourceWidth = photo.getWidth();
        int sourceHeight = photo.getHeight();
        int[] pixels = new int[sourceWidth * sourceHeight];
        photo.getPixels(pixels, 0, sourceWidth, 0, 0, sourceWidth, sourceHeight);
        PhotoTensor.writeRgb(pixels, sourceWidth, sourceHeight, width, height, input);

        float[][] output = new float[1][labels.length + 1];
        model.run(input, output);
        HealthDecision.Result result = healthDecision.decide(output[0]);
        String disease = result.diseaseIndex() < 0 ? null : labels[result.diseaseIndex()];
        return new Prediction(result.outcome(), disease);
    }

    @Override
    public void close() {
        model.close();
    }
}
