from typing import Dict, Any, List
from evidence_retriever import EvidenceDetail

class ConfidenceCalibrator:
    """Calibrates output confidence using weighted combination of rule score, evidence score, personalization, and LLM agreement."""

    def calibrate_confidence(
        self,
        rule_result: Dict[str, Any],
        evidence_items: List[EvidenceDetail],
        features: Dict[str, Any],
        llm_result: Dict[str, Any]
    ) -> Dict[str, Any]:

        # Component 1: Base Rule / LLM Confidence (0.4 weight)
        llm_conf = float(llm_result.get('confidence', 0.70))
        rule_conf = float(rule_result.get('confidence', 0.80)) if rule_result else 0.50
        base_score = 0.6 * llm_conf + 0.4 * rule_conf

        # Component 2: Evidence Strength (0.2 weight)
        top_ev_score = evidence_items[0].total_score if evidence_items else 0.0
        evidence_factor = min(1.0, top_ev_score / 10.0)

        # Component 3: Personalization Clarity (0.2 weight)
        personalization_factor = 0.5
        if features.get('is_dnd') or features.get('is_opted_out') or features.get('is_group_muted'):
            personalization_factor = 0.9
        elif features.get('is_direct_mention') or features.get('sender_is_admin'):
            personalization_factor = 0.95

        # Component 4: Rule & LLM Agreement (0.2 weight)
        agreement_factor = 0.5
        if rule_result:
            if rule_result.get('action') == llm_result.get('action'):
                agreement_factor = 1.0
            else:
                agreement_factor = 0.2

        # Weighted Sum
        final_conf = (
            0.40 * base_score +
            0.20 * evidence_factor +
            0.20 * personalization_factor +
            0.20 * agreement_factor
        )

        # Clamp to [0.0, 1.0]
        clamped_conf = max(0.10, min(0.99, round(final_conf, 2)))

        component_scores = {
            "base_score": round(base_score, 3),
            "evidence_factor": round(evidence_factor, 3),
            "personalization_factor": round(personalization_factor, 3),
            "agreement_factor": round(agreement_factor, 3),
            "calibrated_confidence": clamped_conf
        }

        return component_scores
