import os
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger("MessageRouter")

class DecisionTraceLogger:
    """Logs detailed decision traces for every prediction as JSONL."""

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
            "features": features,
            "rules_triggered": rules_triggered,
            "candidate_evidence_count": len(candidate_evidence),
            "selected_evidence": selected_evidence,
            "baseline_prediction": baseline_prediction,
            "llm_prediction": llm_prediction,
            "confidence_components": confidence_components,
            "final_prediction": final_prediction
        }

        try:
            with open(self.trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_record) + "\n")
        except Exception as e:
            logger.error(f"Failed to log decision trace for {message_id}: {e}")
