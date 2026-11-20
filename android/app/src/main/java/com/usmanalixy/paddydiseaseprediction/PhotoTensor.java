package com.usmanalixy.paddydiseaseprediction;

import java.nio.ByteBuffer;

final class PhotoTensor {
    private PhotoTensor() {}

    static void writeRgb(
            int[] pixels, int width, int height, int targetWidth, int targetHeight, ByteBuffer input) {
        // Match tf.image.resize_with_pad: floor the size and the pre-rounding padding separately.
        float ratio = Math.max((float) width / targetWidth, (float) height / targetHeight);
        float scaledWidth = width / ratio;
        float scaledHeight = height / ratio;
        int left = Math.max(0, (int) ((targetWidth - scaledWidth) / 2));
        int top = Math.max(0, (int) ((targetHeight - scaledHeight) / 2));
        int resizedWidth = Math.max(1, (int) scaledWidth);
        int resizedHeight = Math.max(1, (int) scaledHeight);
        float xScale = (float) width / resizedWidth;
        float yScale = (float) height / resizedHeight;

        input.rewind();
        for (int y = 0; y < targetHeight; y++) {
            for (int x = 0; x < targetWidth; x++) {
                if (x < left || x >= left + resizedWidth || y < top || y >= top + resizedHeight) {
                    input.putFloat(0).putFloat(0).putFloat(0);
                    continue;
                }

                float sourceX = (x - left + 0.5f) * xScale - 0.5f;
                float sourceY = (y - top + 0.5f) * yScale - 0.5f;
                int x0 = (int) Math.floor(sourceX);
                int y0 = (int) Math.floor(sourceY);
                float xf = sourceX - x0;
                float yf = sourceY - y0;
                int row0 = clamp(y0, height) * width;
                int row1 = clamp(y0 + 1, height) * width;
                int col0 = clamp(x0, width);
                int col1 = clamp(x0 + 1, width);
                int a = pixels[row0 + col0];
                int b = pixels[row0 + col1];
                int c = pixels[row1 + col0];
                int d = pixels[row1 + col1];

                input.putFloat(interpolate(a, b, c, d, 16, xf, yf));
                input.putFloat(interpolate(a, b, c, d, 8, xf, yf));
                input.putFloat(interpolate(a, b, c, d, 0, xf, yf));
            }
        }
        input.rewind();
    }

    private static int clamp(int index, int dimension) {
        return Math.max(0, Math.min(index, dimension - 1));
    }

    private static float interpolate(int a, int b, int c, int d, int shift, float x, float y) {
        int topLeft = (a >>> shift) & 255;
        int bottomLeft = (c >>> shift) & 255;
        float top = topLeft + (((b >>> shift) & 255) - topLeft) * x;
        float bottom = bottomLeft + (((d >>> shift) & 255) - bottomLeft) * x;
        return top + (bottom - top) * y;
    }
}
