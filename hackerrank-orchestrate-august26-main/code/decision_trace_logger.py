import os
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger("MessageRouter")

class DecisionTraceLogger:
    """Production Decision Trace Logger generating detailed JSON traces per prediction for debugging and AI Judge interviews."""

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
        trace_record = {
            "message_id": message_id,
            "signals": features,
            "rules_fired": rules_triggered if rules_triggered else "none",
            "evidence_selected": selected_evidence if selected_evidence else [],
            "llm_decision": {
                "action": llm_prediction.get("action", ""),
                "message_type": llm_prediction.get("message_type", ""),
                "reason": llm_prediction.get("reason", ""),
                "confidence": llm_prediction.get("confidence", 0.0)
            },
            "confidence": confidence_components.get("calibrated_confidence", 0.0),
            "confidence_breakdown": confidence_components,
            "final_action": final_prediction.get("action", "")
        }

        try:
            with open(self.trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_record) + "\n")
        except Exception as e:
            logger.error(f"Failed to log decision trace for {message_id}: {e}")
