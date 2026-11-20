package com.usmanalixy.paddydiseaseprediction;

final class HealthDecision {
    enum Outcome { NORMAL, UNCERTAIN, POSSIBLE_DISEASE }

    record Result(Outcome outcome, int diseaseIndex) {}

    private final int normalIndex;
    private final float normalBelow;
    private final float diseaseAbove;
    private final float minDiseaseClassScore;

    HealthDecision(int normalIndex, float normalBelow, float diseaseAbove, float minDiseaseClassScore) {
        if (normalIndex < 0 || !isScore(normalBelow) || !isScore(diseaseAbove)
                || !isScore(minDiseaseClassScore) || normalBelow > diseaseAbove) {
            throw new IllegalArgumentException("Invalid health screen configuration.");
        }
        this.normalIndex = normalIndex;
        this.normalBelow = normalBelow;
        this.diseaseAbove = diseaseAbove;
        this.minDiseaseClassScore = minDiseaseClassScore;
    }

    Result decide(float[] values) {
        int scoreIndex = values.length - 1;
        if (normalIndex >= scoreIndex || !isScore(values[scoreIndex])) {
            throw new IllegalStateException("Invalid health score.");
        }
        int best = 0;
        for (int i = 0; i < scoreIndex; i++) {
            if (!isScore(values[i])) throw new IllegalStateException("Invalid class score.");
            if (values[i] > values[best]) best = i;
        }
        float symptoms = values[scoreIndex];
        if (symptoms < normalBelow) {
            return new Result(Outcome.NORMAL, -1);
        }
        if (symptoms > diseaseAbove && best != normalIndex && values[best] >= minDiseaseClassScore) {
            return new Result(Outcome.POSSIBLE_DISEASE, best);
        }
        return new Result(Outcome.UNCERTAIN, -1);
    }

    private static boolean isScore(float value) {
        return Float.isFinite(value) && value >= 0 && value <= 1;
    }
}
