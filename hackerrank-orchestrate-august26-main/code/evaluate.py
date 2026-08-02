import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from loader import DataLoader
from models import Message
from context_builder import ContextBuilder
from feature_extraction import FeatureExtractor
from evidence_retriever import EvidenceRetriever
from rule_engine import RuleEngine
from llm_reasoner import LLMReasoner
from multimodal_processor import MultimodalProcessor
from confidence_calibrator import ConfidenceCalibrator
from decision_trace_logger import DecisionTraceLogger

class Evaluator:
    """Production Evaluation Framework with Multimodal Processing, Calibrated Confidence, and Decision Trace Logging."""

    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.loader = DataLoader(dataset_dir)
        self.loader.load_all()

        self.context_builder = ContextBuilder(self.loader)
        self.feature_extractor = FeatureExtractor()
        self.evidence_retriever = EvidenceRetriever(self.loader)
        self.rule_engine = RuleEngine()
        self.llm_reasoner = LLMReasoner()
        self.multimodal = MultimodalProcessor(dataset_dir)
        self.calibrator = ConfidenceCalibrator()
        self.trace_logger = DecisionTraceLogger(os.path.join(dataset_dir, "..", "decision_traces.jsonl"))

    def run_evaluation(self):
        sample_path = os.path.join(self.dataset_dir, "sample_messages.csv")
        if not os.path.exists(sample_path):
            print("sample_messages.csv not found.")
            return

        sample_df = pd.read_csv(sample_path).fillna("")
        print(f"Evaluating model against {len(sample_df)} ground truth sample messages...\n")

        y_true_action = []
        y_pred_action = []
        y_true_type = []
        y_pred_type = []
        confidences = []
        evidence_overlaps = []
        errors = []

        for idx, row in sample_df.iterrows():
            msg_dict = row.to_dict()
            msg_id = str(row['message_id'])

            gt_action = str(row['action']).lower().strip()
            gt_type = str(row['message_type']).lower().strip()
            gt_evidence = set([e.strip() for e in str(row['evidence_message_ids']).split(';') if e.strip() and e.strip() != 'none'])

            # Multimodal Enrichment (OCR / Whisper)
            media_type = str(row.get('media_type', ''))
            media_id = str(row.get('media_id', ''))
            msg_text = str(row.get('message_text', ''))
            ocr_result = {}

            if media_type == 'image' and media_id:
                img_path = self.loader.images.get(media_id)
                if img_path:
                    ocr_result = self.multimodal.process_image(img_path.file_path)
                    ocr_text = ocr_result.get('raw_text', '')
                    if ocr_text:
                        msg_text = f"{msg_text} [OCR: {ocr_text}]".strip()
                        msg_dict['message_text'] = msg_text
            elif media_type == 'voice' and media_id:
                vn_path = self.loader.voice_notes.get(media_id)
                if vn_path:
                    asr_text = self.multimodal.process_voice_note(vn_path.file_path)
                    if asr_text:
                        msg_text = asr_text
                        msg_dict['message_text'] = msg_text

            msg_obj = Message(
                message_id=msg_id,
                user_id=str(row['user_id']),
                conversation_type=str(row['conversation_type']),
                group_id=str(row.get('group_id', '')),
                business_id=str(row.get('business_id', '')),
                sender_user_id=str(row.get('sender_user_id', '')),
                created_at=str(row.get('created_at', '')),
                message_text=msg_text,
                media_type=media_type,
                media_id=media_id,
                forwarded_count=int(row.get('forwarded_count', 0) or 0)
            )
            context = self.loader.build_message_context(msg_obj)
            
            legacy_ctx = {
                'message': msg_dict,
                'user': self.loader.users.get(msg_obj.user_id, {}).__dict__ if self.loader.users.get(msg_obj.user_id) else {},
                'group': self.loader.groups.get(msg_obj.group_id, {}).__dict__ if self.loader.groups.get(msg_obj.group_id) else {},
                'group_member': self.loader.group_members.get(f"({msg_obj.group_id},{msg_obj.user_id})", {}).__dict__ if self.loader.group_members.get(f"({msg_obj.group_id},{msg_obj.user_id})") else {},
                'sender_group_member': self.loader.group_members.get(f"({msg_obj.group_id},{msg_obj.sender_user_id})", {}).__dict__ if self.loader.group_members.get(f"({msg_obj.group_id},{msg_obj.sender_user_id})") else {},
                'business': self.loader.businesses.get(msg_obj.business_id, {}).__dict__ if self.loader.businesses.get(msg_obj.business_id) else {},
                'user_business': self.loader.user_business_histories.get(f"({msg_obj.user_id},{msg_obj.business_id})", {}).__dict__ if self.loader.user_business_histories.get(f"({msg_obj.user_id},{msg_obj.business_id})") else {},
                'ocr_meta': ocr_result
            }

            features = self.feature_extractor.extract_features(legacy_ctx)
            evidence_details = self.evidence_retriever.retrieve_evidence_details(legacy_ctx, top_k=3)
            evidence_str = ";".join([e.message_id for e in evidence_details]) if evidence_details else "none"
            pred_evidence_set = set([e.message_id for e in evidence_details])

            rule_result = self.rule_engine.evaluate_rules(legacy_ctx, features, evidence_str)
            
            # Selective Hybrid AI Architecture Optimization:
            # Rule Confidence > 0.90 -> Skip LLM
            # Rule Confidence <= 0.90 -> Call LLM
            rule_conf = float(rule_result.get('confidence', 0.0)) if rule_result else 0.0
            if rule_result and rule_conf > 0.90:
                llm_result = rule_result
            else:
                llm_result = self.llm_reasoner.reason(context, features, rule_result, evidence_details)

            # Confidence Calibration
            calib_scores = self.calibrator.calibrate_confidence(rule_result or {}, evidence_details, features, llm_result)
            final_confidence = calib_scores['calibrated_confidence']
            llm_result['confidence'] = final_confidence

            # Log Decision Trace
            self.trace_logger.log_trace(
                message_id=msg_id,
                features=features,
                rules_triggered=rule_result or {},
                candidate_evidence=[{"id": e.message_id, "score": e.total_score} for e in evidence_details],
                selected_evidence=[e.message_id for e in evidence_details],
                baseline_prediction=rule_result or {},
                llm_prediction=llm_result,
                confidence_components=calib_scores,
                final_prediction=llm_result
            )

            pred_action = str(llm_result['action']).lower().strip()
            pred_type = str(llm_result['message_type']).lower().strip()

            y_true_action.append(gt_action)
            y_pred_action.append(pred_action)
            y_true_type.append(gt_type)
            y_pred_type.append(pred_type)
            confidences.append(final_confidence)

            # Compute Evidence Jaccard Overlap
            if not gt_evidence and not pred_evidence_set:
                overlap = 1.0
            elif not gt_evidence or not pred_evidence_set:
                overlap = 0.0
            else:
                overlap = len(gt_evidence.intersection(pred_evidence_set)) / float(len(gt_evidence.union(pred_evidence_set)))
            evidence_overlaps.append(overlap)

            # Track Mispredictions
            if pred_action != gt_action or pred_type != gt_type:
                errors.append({
                    "message_id": msg_id,
                    "text": str(msg_text)[:80] + "...",
                    "gt_action": gt_action,
                    "pred_action": pred_action,
                    "gt_type": gt_type,
                    "pred_type": pred_type,
                    "gt_evidence": ";".join(gt_evidence) if gt_evidence else "none",
                    "pred_evidence": evidence_str,
                    "features": features,
                    "reason": llm_result.get('reason', '')
                })

        # Calculate Metrics
        action_acc = np.mean([1 if t == p else 0 for t, p in zip(y_true_action, y_pred_action)])
        type_acc = np.mean([1 if t == p else 0 for t, p in zip(y_true_type, y_pred_type)])
        avg_conf = np.mean(confidences)
        avg_evidence_overlap = np.mean(evidence_overlaps)

        # Precision, Recall, F1 & False Counts for Action
        actions = ['notify', 'digest', 'mute']
        precision_dict = {}
        recall_dict = {}
        f1_dict = {}
        false_counts = {}

        for act in actions:
            tp = sum([1 for t, p in zip(y_true_action, y_pred_action) if t == act and p == act])
            fp = sum([1 for t, p in zip(y_true_action, y_pred_action) if t != act and p == act])
            fn = sum([1 for t, p in zip(y_true_action, y_pred_action) if t == act and p != act])
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

            precision_dict[act] = round(p, 4)
            recall_dict[act] = round(r, 4)
            f1_dict[act] = round(f1, 4)
            false_counts[act] = fp

        # Print Clean Evaluation Report
        print("==================================================")
        print("         EVALUATION SUMMARY REPORT               ")
        print("==================================================")
        print(f"Total Sample Messages Evaluated: {len(sample_df)}")
        print(f"Action Accuracy:          {action_acc * 100:.2f}%")
        print(f"Message Type Accuracy:    {type_acc * 100:.2f}%")
        print(f"Average Calibrated Conf:  {avg_conf:.4f} (Clamped [0.35, 0.98])")
        print(f"Evidence Jaccard Overlap: {avg_evidence_overlap * 100:.2f}%\n")

        print("--- FALSE PREDICTION COUNTS ---")
        print(f"False NOTIFY Count: {false_counts['notify']}")
        print(f"False DIGEST Count: {false_counts['digest']}")
        print(f"False MUTE Count:   {false_counts['mute']}\n")

        print("--- PER-ACTION METRICS ---")
        for act in actions:
            print(f"[{act.upper():<6}] Precision: {precision_dict[act]:.4f} | Recall: {recall_dict[act]:.4f} | F1-Score: {f1_dict[act]:.4f}")

        # Confusion Matrix
        print("\n--- CONFUSION MATRIX (Action) ---")
        conf_matrix = pd.crosstab(
            pd.Series(y_true_action, name='Actual'),
            pd.Series(y_pred_action, name='Predicted')
        )
        print(conf_matrix)

        # Top Incorrect Predictions Analysis
        print("\n==================================================")
        print("       TOP INCORRECT PREDICTIONS ANALYSIS         ")
        print("==================================================")
        if not errors:
            print("Zero mispredictions! 100% accuracy achieved on sample set.")
        else:
            for i, err in enumerate(errors[:10], 1):
                print(f"\n{i}. Message ID: {err['message_id']}")
                print(f"   Snippet: \"{err['text']}\"")
                print(f"   Actual:    Action='{err['gt_action']}', Type='{err['gt_type']}', Evidence='{err['gt_evidence']}'")
                print(f"   Predicted: Action='{err['pred_action']}', Type='{err['pred_type']}', Evidence='{err['pred_evidence']}'")
                print(f"   Reason:    {err['reason']}")
                print(f"   Key Signals: DND={err['features'].get('is_dnd')}, OptOut={err['features'].get('is_opted_out')}, MutedGroup={err['features'].get('is_group_muted')}, Mention={err['features'].get('is_direct_mention')}")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_directory = os.path.join(base_dir, "dataset")
    evaluator = Evaluator(dataset_directory)
    evaluator.run_evaluation()
