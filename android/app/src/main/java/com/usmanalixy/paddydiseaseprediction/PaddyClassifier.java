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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

final class PaddyClassifier implements AutoCloseable {
    record Prediction(String label, float score) {}

    private final Interpreter model;
    private final ByteBuffer input;
    private final String[] labels;
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
                    || !Arrays.equals(modelOutput.shape(), new int[] {1, labels.length})) {
                throw new IOException("Model tensors do not match the bundled metadata.");
            }
        } catch (IOException | RuntimeException error) {
            model.close();
            throw error;
        }
    }

    List<Prediction> predict(Bitmap photo) {
        int sourceWidth = photo.getWidth();
        int sourceHeight = photo.getHeight();
        int[] pixels = new int[sourceWidth * sourceHeight];
        photo.getPixels(pixels, 0, sourceWidth, 0, 0, sourceWidth, sourceHeight);
        PhotoTensor.writeRgb(pixels, sourceWidth, sourceHeight, width, height, input);

        float[][] output = new float[1][labels.length];
        model.run(input, output);
        List<Prediction> scores = new ArrayList<>(labels.length);
        for (int i = 0; i < labels.length; i++) {
            float score = output[0][i];
            if (!Float.isFinite(score) || score < 0 || score > 1) {
                throw new IllegalStateException("Invalid classifier output.");
            }
            scores.add(new Prediction(labels[i], score));
        }
        scores.sort(Comparator.comparingDouble(Prediction::score).reversed());
        return scores.subList(0, Math.min(3, scores.size()));
    }

    @Override
    public void close() {
        model.close();
    }
}
