package com.usmanalixy.paddydiseaseprediction;

import static org.junit.Assert.assertArrayEquals;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import org.junit.Test;

public final class PhotoTensorTest {
    @Test
    public void writesRgbWithBilinearInterpolationAndBlackPadding() {
        float[] result = resize(new int[] {0xFFFF0000, 0xFF0000FF}, 2, 1, 4, 4);
        assertArrayEquals(new float[] {
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                255, 0, 0, 191.25f, 0, 63.75f, 63.75f, 0, 191.25f, 0, 0, 255,
                255, 0, 0, 191.25f, 0, 63.75f, 63.75f, 0, 191.25f, 0, 0, 255,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        }, result, 0.001f);
    }

    @Test
    public void roundsPaddingBeforeTruncatingResizedDimensionsLikeTensorFlow() {
        int[] pixels = {
                0xFF00FF00, 0xFF00FF00, 0xFF00FF00,
                0xFF00FF00, 0xFF00FF00, 0xFF00FF00
        };
        float[] result = resize(pixels, 3, 2, 5, 5);
        float[] expected = new float[5 * 5 * 3];
        for (int pixel = 0; pixel < 3 * 5; pixel++) expected[pixel * 3 + 1] = 255;
        assertArrayEquals(expected, result, 0.001f);
    }

    private static float[] resize(int[] pixels, int width, int height, int targetWidth, int targetHeight) {
        float[] result = new float[targetWidth * targetHeight * 3];
        ByteBuffer input = ByteBuffer.allocate(result.length * Float.BYTES)
                .order(ByteOrder.nativeOrder());
        PhotoTensor.writeRgb(pixels, width, height, targetWidth, targetHeight, input);
        input.asFloatBuffer().get(result);
        return result;
    }
}
