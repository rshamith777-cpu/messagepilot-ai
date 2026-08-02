from typing import Dict, Any, List
from evidence_retriever import EvidenceDetail

class ConfidenceCalibrator:
    """
    Production Confidence Calibration Module.
    
    Formula:
      Calibrated Confidence = (0.40 * Rule Certainty)
                            + (0.30 * Evidence Strength)
                            + (0.20 * Personalization Factor)
                            + (0.10 * LLM Certainty)
                            
    Clamped strictly between [0.35, 0.98].
    """

    def calibrate_confidence(
        self,
        rule_result: Dict[str, Any],
        evidence_items: List[EvidenceDetail],
        features: Dict[str, Any],
        llm_result: Dict[str, Any]
    ) -> Dict[str, Any]:

        # 1. Rule Certainty (40% Weight)
        rule_certainty = float(rule_result.get('confidence', 0.85)) if rule_result else 0.50

        # 2. Evidence Strength (30% Weight)
        top_ev_score = evidence_items[0].total_score if evidence_items else 0.0
        evidence_strength = min(1.0, top_ev_score / 10.0)

        # 3. Personalization Clarity (20% Weight)
        personalization_factor = 0.50
        if features.get('is_dnd') or features.get('is_opted_out') or features.get('is_group_muted'):
            personalization_factor = 0.90
        elif features.get('is_direct_mention') or features.get('sender_is_admin'):
            personalization_factor = 0.95

        # 4. LLM Certainty (10% Weight)
        llm_certainty = float(llm_result.get('confidence', 0.75)) if llm_result else 0.50

        # Weighted Calibration Formula
        raw_calibrated = (
            (0.40 * rule_certainty) +
            (0.30 * evidence_strength) +
            (0.20 * personalization_factor) +
            (0.10 * llm_certainty)
        )

        # Clamp strictly between 0.35 and 0.98
        clamped_conf = max(0.35, min(0.98, round(raw_calibrated, 2)))

        return {
            "rule_certainty": round(rule_certainty, 3),
            "evidence_strength": round(evidence_strength, 3),
            "personalization_factor": round(personalization_factor, 3),
            "llm_certainty": round(llm_certainty, 3),
            "raw_calibrated": round(raw_calibrated, 4),
            "calibrated_confidence": clamped_conf
        }
