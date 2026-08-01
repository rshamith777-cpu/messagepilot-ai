import re
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from data_loader import DataLoader

# ==========================================
# CONFIGURABLE SCORING WEIGHTS CONSTANTS
# ==========================================
WEIGHT_BUSINESS_MATCH = 5.0
WEIGHT_GROUP_MATCH = 3.0
WEIGHT_SAME_SENDER = 2.0
WEIGHT_WORD_OVERLAP = 0.5

# Event & Historical Engagement Weights
WEIGHT_EVENT_REPORTED = 10.0
WEIGHT_EVENT_MUTED = 4.0
WEIGHT_EVENT_REPLIED = 2.0
WEIGHT_EVENT_OPENED = 1.0
WEIGHT_EVENT_DISMISSED = -1.5  # Negative score for dismissed noise

# Duplicate Detection Weights
WEIGHT_EXACT_DUPLICATE = 8.0
WEIGHT_NEAR_DUPLICATE = 4.0

# Recency Decay Constant (decay per 30 days)
RECENCY_DECAY_HALF_LIFE_DAYS = 30.0
MAX_RECENCY_BONUS = 3.0

@dataclass
class EvidenceDetail:
    message_id: str
    total_score: float
    score_breakdown: Dict[str, float]
    triggered_signals: List[str]

class EvidenceRetriever:
    """Enhanced Dedicated Evidence Retrieval Module:
    1. Configurable Scoring Constants
    2. Candidate Retrieval with Recency Scoring (exponential decay)
    3. Duplicate Detection (Exact string match & Jaccard near-duplicate match)
    4. Full Historical Engagement Integration (opened, replied, dismissed, muted, reported)
    5. Rich Evidence Objects (message_id, total_score, score_breakdown, triggered_signals)
    """

    def __init__(self, data_loader: DataLoader):
        self.dl = data_loader
        self.events_map: Dict[str, Dict[str, Any]] = {}
        if hasattr(self.dl, 'message_events_df') and self.dl.message_events_df is not None:
            for _, row in self.dl.message_events_df.iterrows():
                m_id = str(row['message_id'])
                u_id = str(row['user_id'])
                self.events_map[f"{u_id}_{m_id}"] = row.to_dict()

    def _compute_jaccard_similarity(self, set1: set, set2: set) -> float:
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def _calculate_recency_score(self, current_time_str: str, cand_time_str: str) -> Tuple[float, Optional[float]]:
        if not current_time_str or not cand_time_str:
            return 0.0, None
        try:
            curr_dt = datetime.strptime(current_time_str, "%Y-%m-%d %H:%M")
            cand_dt = datetime.strptime(cand_time_str, "%Y-%m-%d %H:%M")
            days_diff = max(0.0, (curr_dt - cand_dt).total_seconds() / 86400.0)
            
            # Recency bonus with exponential decay
            score = MAX_RECENCY_BONUS * (0.5 ** (days_diff / RECENCY_DECAY_HALF_LIFE_DAYS))
            return round(score, 3), days_diff
        except Exception:
            return 0.0, None

    def retrieve_evidence_details(self, context: Dict[str, Any], top_k: int = 3) -> List[EvidenceDetail]:
        msg = context['message']
        user_id = str(msg.get('user_id', ''))
        sender_id = str(msg.get('sender_user_id', ''))
        group_id = str(msg.get('group_id', ''))
        business_id = str(msg.get('business_id', ''))
        created_at_str = str(msg.get('created_at', ''))
        msg_text = str(msg.get('message_text', '')).strip().lower()
        msg_words = set(re.findall(r'\w+', msg_text))

        # 1. Candidate Retrieval
        candidates = []
        msg_hist = getattr(self.dl, 'message_history_df', None)
        if msg_hist is None:
            msg_hist = getattr(self.dl, 'message_history', [])

        if isinstance(msg_hist, list):
            for item in msg_hist:
                row = item.__dict__ if hasattr(item, '__dict__') else item
                cand_user = str(row.get('user_id', ''))
                if cand_user != user_id:
                    continue
                cand_group = str(row.get('group_id', ''))
                cand_biz = str(row.get('business_id', ''))
                cand_sender = str(row.get('sender_user_id', ''))
                if (group_id and cand_group == group_id) or \
                   (business_id and cand_biz == business_id) or \
                   (sender_id and cand_sender == sender_id):
                    candidates.append(row)
        elif hasattr(msg_hist, 'iterrows'):
            for _, row_series in msg_hist.iterrows():
                row = row_series.to_dict()
                cand_user = str(row.get('user_id', ''))
                if cand_user != user_id:
                    continue
                cand_group = str(row.get('group_id', ''))
                cand_biz = str(row.get('business_id', ''))
                cand_sender = str(row.get('sender_user_id', ''))
                if (group_id and cand_group == group_id) or \
                   (business_id and cand_biz == business_id) or \
                   (sender_id and cand_sender == sender_id):
                    candidates.append(row)

        if not candidates:
            return []

        # 2. Additive Feature Scoring with Detailed Breakdown
        evidence_results: List[EvidenceDetail] = []

        for cand in candidates:
            cand_id = str(cand['message_id'])
            cand_text = str(cand.get('message_text', '')).strip().lower()
            cand_words = set(re.findall(r'\w+', cand_text))
            cand_time_str = str(cand.get('created_at', ''))

            breakdown: Dict[str, float] = {}
            signals: List[str] = []

            # Entity Match
            if business_id and str(cand.get('business_id', '')) == business_id:
                breakdown['business_match'] = WEIGHT_BUSINESS_MATCH
                signals.append("same_business")
            if group_id and str(cand.get('group_id', '')) == group_id:
                breakdown['group_match'] = WEIGHT_GROUP_MATCH
                signals.append("same_group")
                if sender_id and str(cand.get('sender_user_id', '')) == sender_id:
                    breakdown['same_sender'] = WEIGHT_SAME_SENDER
                    signals.append("same_sender")

            # Duplicate Detection
            if msg_text and cand_text and msg_text == cand_text:
                breakdown['exact_duplicate'] = WEIGHT_EXACT_DUPLICATE
                signals.append("exact_duplicate")
            else:
                jaccard = self._compute_jaccard_similarity(msg_words, cand_words)
                if jaccard >= 0.7:
                    breakdown['near_duplicate'] = WEIGHT_NEAR_DUPLICATE
                    signals.append(f"near_duplicate(similarity={jaccard:.2f})")

            # Word Overlap
            overlap = len(msg_words.intersection(cand_words))
            if overlap > 0:
                breakdown['word_overlap'] = round(overlap * WEIGHT_WORD_OVERLAP, 2)
                signals.append(f"word_overlap({overlap})")

            # Recency Scoring
            recency_score, days_ago = self._calculate_recency_score(created_at_str, cand_time_str)
            if recency_score > 0:
                breakdown['recency'] = recency_score
                if days_ago is not None:
                    signals.append(f"recency({days_ago:.1f}d_ago)")

            # Historical Engagement Reactions
            event = self.events_map.get(f"{user_id}_{cand_id}", {})
            if event:
                if event.get('message_reported', 0) in [1, "1"]:
                    breakdown['event_reported'] = WEIGHT_EVENT_REPORTED
                    signals.append("user_reported")
                if event.get('muted_after_message', 0) in [1, "1"]:
                    breakdown['event_muted'] = WEIGHT_EVENT_MUTED
                    signals.append("muted_after_message")
                if event.get('message_replied', 0) in [1, "1"]:
                    breakdown['event_replied'] = WEIGHT_EVENT_REPLIED
                    signals.append("user_replied")
                if event.get('message_opened', 0) in [1, "1"]:
                    breakdown['event_opened'] = WEIGHT_EVENT_OPENED
                    signals.append("user_opened")
                if event.get('notification_dismissed', 0) in [1, "1"]:
                    breakdown['event_dismissed'] = WEIGHT_EVENT_DISMISSED
                    signals.append("user_dismissed")

            total_score = round(sum(breakdown.values()), 3)

            if total_score > 1.0:
                evidence_results.append(EvidenceDetail(
                    message_id=cand_id,
                    total_score=total_score,
                    score_breakdown=breakdown,
                    triggered_signals=signals
                ))

        # Sort descending by total_score
        evidence_results.sort(key=lambda x: x.total_score, reverse=True)
        return evidence_results[:top_k]

    def retrieve_evidence(self, context: Dict[str, Any], top_k: int = 3) -> str:
        """Pipeline compatibility method returning semicolon-separated string or 'none'."""
        details = self.retrieve_evidence_details(context, top_k=top_k)
        if not details:
            return "none"
        return ";".join([d.message_id for d in details])
