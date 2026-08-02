import os
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger("MessageRouter")

class DecisionTraceLogger:
    """Production Decision Graph Logger generating explainable JSON graph traces per prediction for debugging and AI Judge interviews."""

    def __init__(self, trace_path: str = "decision_traces.jsonl"):
        self.trace_path = trace_path

    def log_trace(
        self,
        message_id: str,
        features: Dict[str, Any],
        rules_triggered: Dict[str, Any],
        candidate_evidence: List[Dict[str, Any]],
        selected_evidence: List[str],
        baseline_prediction: Dict[str, Any],
        llm_prediction: Dict[str, Any],
        confidence_components: Dict[str, Any],
        final_prediction: Dict[str, Any]
    ):
        scorecard = rules_triggered.get('scorecard', {'notify': 0.0, 'digest': 0.0, 'mute': 0.0}) if isinstance(rules_triggered, dict) else {'notify': 0.0, 'digest': 0.0, 'mute': 0.0}
        triggered_signals = rules_triggered.get('triggered_signals', []) if isinstance(rules_triggered, dict) else []

        trace_record = {
            "message_id": message_id,
            "decision_graph": {
                "extracted_signals": features,
                "scorecard_vector": scorecard,
                "triggered_signals": triggered_signals,
                "candidate_evidence": candidate_evidence,
                "selected_evidence_ids": selected_evidence if selected_evidence else [],
                "baseline_recommendation": baseline_prediction if baseline_prediction else {},
                "llm_reasoning_node": {
                    "action": llm_prediction.get("action", ""),
                    "message_type": llm_prediction.get("message_type", ""),
                    "reason": llm_prediction.get("reason", ""),
                    "confidence": llm_prediction.get("confidence", 0.0)
                },
                "confidence_calibration_node": confidence_components,
                "final_prediction_node": {
                    "action": final_prediction.get("action", ""),
                    "message_type": final_prediction.get("message_type", ""),
                    "reason": final_prediction.get("reason", ""),
                    "confidence": confidence_components.get("calibrated_confidence", 0.0)
                }
            }
        }

        try:
            with open(self.trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_record) + "\n")
        except Exception as e:
            logger.error(f"Failed to log decision trace for {message_id}: {e}")
