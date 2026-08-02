import os
import sys
import time
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
from user_memory_engine import UserMemoryEngine

def run_large_scale_analytics():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_1000_benchmarks.csv")
    if not os.path.exists(csv_path):
        print(f"Synthetic benchmark file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path).head(250)
    loader = DataLoader(os.path.join(BASE_DIR, "dataset"))
    loader.load_all()

    fe = FeatureExtractor()
    re = RuleEngine()
    er = EvidenceRetriever(loader)
    cc = ConfidenceCalibrator()
    ume = UserMemoryEngine(loader)

    y_true_act = []
    y_pred_act = []
    y_true_type = []
    y_pred_type = []
    confidences = []
    latencies_ms = []

    print("\n==========================================================================")
    print("      LARGE-SCALE BENCHMARK & SYSTEMS ANALYTICS ENGINE (250 MESSAGES)     ")
    print("==========================================================================\n")

    t_start_total = time.time()

    for _, row in df.iterrows():
        t0 = time.time()
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
            'business': {'official_domain': 'bank.com' if 'OTP' in text else 'amazon.in'},
            'user_business': {'allows_promotions': 0 if 'STOP' in text else 1},
            'ocr_meta': {}
        }

        feats = fe.extract_features(dummy_ctx)
        ev_details = er.retrieve_evidence_details(dummy_ctx, top_k=2)
        ev_str = ";".join([e.message_id for e in ev_details]) if ev_details else "none"

        mem_mods = ume.get_user_personalization_modifier('u_001')
        rule_res = re.evaluate_rules(dummy_ctx, feats, ev_str)
        
        calib = cc.calibrate_confidence(rule_res or {}, ev_details, feats, rule_res or {})
        conf = calib['calibrated_confidence']

        pred_act = str(rule_res['action']).lower() if rule_res else "digest"
        pred_type = str(rule_res['message_type']).lower() if rule_res else "personal"

        t1 = time.time()
        latencies_ms.append((t1 - t0) * 1000.0)

        y_true_act.append(gt_act)
        y_pred_act.append(pred_act)
        y_true_type.append(gt_type)
        y_pred_type.append(pred_type)
        confidences.append(conf)

    t_end_total = time.time()

    total_time = t_end_total - t_start_total
    throughput = len(df) / total_time if total_time > 0 else 0.0

    act_acc = np.mean([1 if t == p else 0 for t, p in zip(y_true_act, y_pred_act)])
    type_acc = np.mean([1 if t == p else 0 for t, p in zip(y_true_type, y_pred_type)])
    avg_conf = np.mean(confidences)

    print("==========================================================================")
    print("                   SYSTEM PERFORMANCE & METRICS REPORT                    ")
    print("==========================================================================")
    print(f"Total Benchmark Size:       {len(df)} Messages")
    print(f"Total Execution Time:       {total_time:.3f} Seconds")
    print(f"Pipeline Throughput:        {throughput:.1f} Messages / Second")
    print(f"Average Latency per Msg:    {np.mean(latencies_ms):.2f} ms (p95: {np.percentile(latencies_ms, 95):.2f} ms)")
    print(f"Action Accuracy:            {act_acc * 100:.2f}%")
    print(f"Message Type Accuracy:      {type_acc * 100:.2f}%")
    print(f"Mean Calibrated Conf:       {avg_conf:.4f} (Min: {np.min(confidences):.2f}, Max: {np.max(confidences):.2f})")
    
    # False Prediction Counts
    false_notify = sum([1 for t, p in zip(y_true_act, y_pred_act) if p == 'notify' and t != 'notify'])
    false_digest = sum([1 for t, p in zip(y_true_act, y_pred_act) if p == 'digest' and t != 'digest'])
    false_mute = sum([1 for t, p in zip(y_true_act, y_pred_act) if p == 'mute' and t != 'mute'])

    print("\n--- ERROR ANALYSIS ---")
    print(f"False NOTIFY Count: {false_notify}")
    print(f"False DIGEST Count: {false_digest}")
    print(f"False MUTE Count:   {false_mute}")
    print("==========================================================================\n")

if __name__ == "__main__":
    run_large_scale_analytics()
