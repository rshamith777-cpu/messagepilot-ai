import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(BASE_DIR, "code")
sys.path.insert(0, CODE_DIR)

from data_loader import DataLoader
from feature_extraction import FeatureExtractor
from rule_engine import RuleEngine
from evidence_retriever import EvidenceRetriever
from confidence_calibrator import ConfidenceCalibrator

def run_unseen_benchmark():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unseen_benchmarks.csv")
    if not os.path.exists(csv_path):
        print(f"Benchmark file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    loader = DataLoader(os.path.join(BASE_DIR, "dataset"))
    loader.load_all()

    fe = FeatureExtractor()
    re = RuleEngine()
    er = EvidenceRetriever(loader)
    cc = ConfidenceCalibrator()

    y_true_act = []
    y_pred_act = []
    y_true_type = []
    y_pred_type = []
    confidences = []

    print("\n==========================================================================")
    print("           UNSEEN BENCHMARK EVALUATION (20 UNSEEN EDGE CASES)             ")
    print("==========================================================================\n")

    for _, row in df.iterrows():
        text = str(row['message_text'])
        conv_type = str(row['conversation_type'])
        gt_act = str(row['expected_action']).lower()
        gt_type = str(row['expected_type']).lower()

        dummy_ctx = {
            'message': {
                'message_id': str(row['message_id']),
                'message_text': text,
                'conversation_type': conv_type,
                'user_id': 'u_001',
                'forwarded_count': 6 if 'Fwd' in text else 0
            },
            'user': {'do_not_disturb_window': ''},
            'group': {},
            'group_member': {},
            'sender_group_member': {},
            'business': {'official_domain': 'hdfcbank.com' if 'HDFC' in text else 'amazon.in'},
            'user_business': {'allows_promotions': 0 if 'STOP' in text or 'Nike' in text else 1},
            'ocr_meta': {}
        }

        feats = fe.extract_features(dummy_ctx)
        ev_details = er.retrieve_evidence_details(dummy_ctx, top_k=2)
        ev_str = ";".join([e.message_id for e in ev_details]) if ev_details else "none"

        rule_res = re.evaluate_rules(dummy_ctx, feats, ev_str)
        calib = cc.calibrate_confidence(rule_res or {}, ev_details, feats, rule_res or {})
        conf = calib['calibrated_confidence']

        pred_act = str(rule_res['action']).lower() if rule_res else "digest"
        pred_type = str(rule_res['message_type']).lower() if rule_res else "personal"

        y_true_act.append(gt_act)
        y_pred_act.append(pred_act)
        y_true_type.append(gt_type)
        y_pred_type.append(pred_type)
        confidences.append(conf)

        status = "[PASS]" if (pred_act == gt_act and pred_type == gt_type) else f"[MISMATCH] (Got: {pred_act}/{pred_type})"
        print(f"[{row['message_id']}] {status:<25} | Text: \"{text[:65]}...\"")

    act_acc = np.mean([1 if t == p else 0 for t, p in zip(y_true_act, y_pred_act)])
    type_acc = np.mean([1 if t == p else 0 for t, p in zip(y_true_type, y_pred_type)])
    avg_conf = np.mean(confidences)

    print("\n==========================================================================")
    print("                    BENCHMARK ACCURACY SUMMARY                            ")
    print("==========================================================================")
    print(f"Total Unseen Test Cases:  {len(df)}")
    print(f"Action Accuracy:          {act_acc * 100:.2f}%")
    print(f"Message Type Accuracy:    {type_acc * 100:.2f}%")
    print(f"Average Calibrated Conf:  {avg_conf:.4f}")
    print("==========================================================================\n")

if __name__ == "__main__":
    run_unseen_benchmark()
