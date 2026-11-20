package com.usmanalixy.paddydiseaseprediction;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

import org.junit.Test;

public final class HealthDecisionTest {
    private final HealthDecision decision = new HealthDecision(0, 0.2f, 0.8f, 0.6f);

    @Test
    public void strongNormalEvidenceSuppressesAllDiseaseLabels() {
        assertEquals(new HealthDecision.Result(HealthDecision.Outcome.NORMAL, -1),
                decision.decide(new float[] {0.1f, 0.9f, 0.1f}));
    }

    @Test
    public void highSymptomAndDiseaseScoresReportOnlyTheLeadingDisease() {
        assertEquals(new HealthDecision.Result(HealthDecision.Outcome.POSSIBLE_DISEASE, 2),
                decision.decide(new float[] {0.1f, 0.2f, 0.7f, 0.95f}));
    }

    @Test
    public void uncertainAndConflictingScoresDoNotShowDiseaseLabels() {
        HealthDecision.Result uncertain = new HealthDecision.Result(HealthDecision.Outcome.UNCERTAIN, -1);
        assertEquals(uncertain, decision.decide(new float[] {0.1f, 0.9f, 0.2f}));
        assertEquals(uncertain, decision.decide(new float[] {0.1f, 0.9f, 0.8f}));
        assertEquals(uncertain, decision.decide(new float[] {0.3f, 0.5f, 0.2f, 0.99f}));
        assertEquals(uncertain, decision.decide(new float[] {0.9f, 0.1f, 0.99f}));
    }

    @Test
    public void invalidModelValuesAreRejected() {
        assertThrows(IllegalStateException.class, () -> decision.decide(new float[] {0.2f, 0.8f, Float.NaN}));
        assertThrows(IllegalStateException.class, () -> decision.decide(new float[] {Float.NaN, 0.8f, 0.1f}));
        assertThrows(IllegalStateException.class, () -> decision.decide(new float[0]));
    }
}
