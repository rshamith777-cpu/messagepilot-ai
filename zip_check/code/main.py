import os
import pandas as pd
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

def process_messages(dataset_dir: str, output_path: str):
    # 1. Initialize Pipeline Components
    loader = DataLoader(dataset_dir)
    loader.load_all()

    context_builder = ContextBuilder(loader)
    feature_extractor = FeatureExtractor()
    evidence_retriever = EvidenceRetriever(loader)
    rule_engine = RuleEngine()
    llm_reasoner = LLMReasoner()
    multimodal = MultimodalProcessor(dataset_dir)
    calibrator = ConfidenceCalibrator()
    
    trace_path = os.path.join(dataset_dir, "..", "decision_traces.jsonl")
    try:
        if os.path.exists(trace_path):
            with open(trace_path, "w", encoding="utf-8") as f:
                f.truncate(0)
    except Exception:
        pass
    trace_logger = DecisionTraceLogger(trace_path)

    results = []

    # 2. Process and Route Each Incoming Message in messages.csv (loader.messages is List[Message])
    for msg_obj in loader.messages:
        msg_id = msg_obj.message_id
        media_type = msg_obj.media_type
        media_id = msg_obj.media_id
        msg_text = msg_obj.message_text

        msg_dict = {
            'message_id': msg_obj.message_id,
            'user_id': msg_obj.user_id,
            'conversation_type': msg_obj.conversation_type,
            'group_id': msg_obj.group_id,
            'business_id': msg_obj.business_id,
            'sender_user_id': msg_obj.sender_user_id,
            'created_at': msg_obj.created_at,
            'message_text': msg_text,
            'media_type': media_type,
            'media_id': media_id,
            'forwarded_count': msg_obj.forwarded_count
        }

        # Step 1: Multimodal Pre-Processing (OCR / Speech-to-Text)
        ocr_result = {}
        if media_type == 'image' and media_id:
            img_info = loader.images.get(media_id)
            if img_info:
                ocr_result = multimodal.process_image(img_info.file_path)
                ocr_text = ocr_result.get('raw_text', '')
                if ocr_text:
                    msg_text = f"{msg_text} [OCR: {ocr_text}]".strip()
                    msg_dict['message_text'] = msg_text
                    msg_obj.message_text = msg_text
        elif media_type == 'voice' and media_id:
            vn_info = loader.voice_notes.get(media_id)
            if vn_info:
                asr_text = multimodal.process_voice_note(vn_info.file_path)
                if asr_text:
                    msg_text = asr_text
                    msg_dict['message_text'] = msg_text
                    msg_obj.message_text = msg_text

        # Step 2: Build Typed Context
        context = loader.build_message_context(msg_obj)

        legacy_ctx = {
            'message': msg_dict,
            'user': loader.users.get(msg_obj.user_id, {}).__dict__ if loader.users.get(msg_obj.user_id) else {},
            'group': loader.groups.get(msg_obj.group_id, {}).__dict__ if loader.groups.get(msg_obj.group_id) else {},
            'group_member': loader.group_members.get(f"({msg_obj.group_id},{msg_obj.user_id})", {}).__dict__ if loader.group_members.get(f"({msg_obj.group_id},{msg_obj.user_id})") else {},
            'sender_group_member': loader.group_members.get(f"({msg_obj.group_id},{msg_obj.sender_user_id})", {}).__dict__ if loader.group_members.get(f"({msg_obj.group_id},{msg_obj.sender_user_id})") else {},
            'business': loader.businesses.get(msg_obj.business_id, {}).__dict__ if loader.businesses.get(msg_obj.business_id) else {},
            'user_business': loader.user_business_histories.get(f"({msg_obj.user_id},{msg_obj.business_id})", {}).__dict__ if loader.user_business_histories.get(f"({msg_obj.user_id},{msg_obj.business_id})") else {},
            'ocr_meta': ocr_result
        }

        # Step 3: Extract Features
        features = feature_extractor.extract_features(legacy_ctx)

        # Step 4: Evidence Retrieval
        evidence_details = evidence_retriever.retrieve_evidence_details(legacy_ctx, top_k=3)
        evidence_str = ";".join([e.message_id for e in evidence_details]) if evidence_details else "none"

        # Step 5: High-Precision Rule Engine
        rule_result = rule_engine.evaluate_rules(legacy_ctx, features, evidence_str)

        # Step 6: Selective Hybrid Reasoning Optimization
        # Rule Confidence > 0.90 -> Skip LLM
        # Rule Confidence <= 0.90 -> Call LLM
        rule_conf = float(rule_result.get('confidence', 0.0)) if rule_result else 0.0
        if rule_result and rule_conf > 0.90:
            decision = rule_result
        else:
            decision = llm_reasoner.reason(context, features, rule_result, evidence_details)

        # Step 7: Confidence Calibration
        calib_scores = calibrator.calibrate_confidence(rule_result or {}, evidence_details, features, decision)
        final_confidence = calib_scores['calibrated_confidence']

        # Step 8: Log Decision Trace
        trace_logger.log_trace(
            message_id=msg_id,
            features=features,
            rules_triggered=rule_result or {},
            candidate_evidence=[{"id": e.message_id, "score": e.total_score} for e in evidence_details],
            selected_evidence=[e.message_id for e in evidence_details],
            baseline_prediction=rule_result or {},
            llm_prediction=decision,
            confidence_components=calib_scores,
            final_prediction=decision
        )

        ev_val = decision.get('evidence_message_ids', 'none')
        if isinstance(ev_val, list):
            ev_str = ";".join([str(e).strip() for e in ev_val if str(e).strip() and str(e).strip() != 'none'])
            if not ev_str:
                ev_str = "none"
        else:
            ev_str = str(ev_val).strip() if str(ev_val).strip() else "none"

        results.append({
            'message_id': msg_id,
            'action': str(decision['action']).lower().strip(),
            'message_type': str(decision['message_type']).lower().strip(),
            'reason': str(decision['reason']).strip(),
            'confidence': f"{final_confidence:.2f}",
            'evidence_message_ids': ev_str
        })

    # Step 9: Write exact output schema to output.csv
    output_df = pd.DataFrame(results, columns=['message_id', 'action', 'message_type', 'reason', 'confidence', 'evidence_message_ids'])
    output_df.to_csv(output_path, index=False)
    print(f"Successfully generated {len(output_df)} predictions in {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_directory = os.path.join(base_dir, "dataset")
    output_csv_path = os.path.join(dataset_directory, "output.csv")
    process_messages(dataset_directory, output_csv_path)
