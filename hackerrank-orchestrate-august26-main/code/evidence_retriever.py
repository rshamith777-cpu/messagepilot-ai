import re
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from data_loader import DataLoader

WEIGHT_BUSINESS_MATCH = 4.0
WEIGHT_GROUP_MATCH = 3.0
WEIGHT_SAME_SENDER = 2.0
WEIGHT_WORD_OVERLAP = 2.0

WEIGHT_EVENT_REPORTED = 10.0
WEIGHT_EVENT_MUTED = 4.0
WEIGHT_EVENT_REPLIED = 3.0
WEIGHT_EVENT_OPENED = 1.0

WEIGHT_EXACT_DUPLICATE = 15.0
WEIGHT_NEAR_DUPLICATE = 8.0

RECENCY_DECAY_HALF_LIFE_DAYS = 90.0
MAX_RECENCY_BONUS = 2.0

@dataclass
class EvidenceDetail:
    message_id: str
    total_score: float
    score_breakdown: Dict[str, float]
    triggered_signals: List[str]

class EvidenceRetriever:
    """Enhanced Evidence Retrieval Module with topic matching, entity scoring, and strict precision ranking."""

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
        
        stop_words = {'the', 'a', 'an', 'and', 'or', 'is', 'in', 'at', 'of', 'to', 'for', 'with', 'on', 'this', 'that', 'it', 'you', 'your', 'i', 'we', 'my', 'be', 'are', 'have', 'has', 'pls', 'please'}
        msg_words = set([w for w in re.findall(r'\w+', msg_text) if len(w) > 2 and w not in stop_words])

        candidates = []
        msg_hist = getattr(self.dl, 'message_history_df', None)
        if msg_hist is None:
            msg_hist = getattr(self.dl, 'message_history', [])

        if isinstance(msg_hist, list):
            for item in msg_hist:
                row = item.__dict__ if hasattr(item, '__dict__') else item
                if str(row.get('user_id', '')) != user_id:
                    continue
                candidates.append(row)
        elif hasattr(msg_hist, 'iterrows'):
            for _, row_series in msg_hist.iterrows():
                row = row_series.to_dict()
                if str(row.get('user_id', '')) != user_id:
                    continue
                candidates.append(row)

        if not candidates:
            return []

        evidence_results: List[EvidenceDetail] = []

        for cand in candidates:
            cand_id = str(cand['message_id'])
            cand_text = str(cand.get('message_text', '')).strip().lower()
            cand_words = set([w for w in re.findall(r'\w+', cand_text) if len(w) > 2 and w not in stop_words])
            cand_time_str = str(cand.get('created_at', ''))

            breakdown: Dict[str, float] = {}
            signals: List[str] = []

            # Entity Match Signals
            if business_id and str(cand.get('business_id', '')) == business_id:
                breakdown['business_match'] = WEIGHT_BUSINESS_MATCH
                signals.append("same_business")
            if group_id and str(cand.get('group_id', '')) == group_id:
                breakdown['group_match'] = WEIGHT_GROUP_MATCH
                signals.append("same_group")
            if sender_id and str(cand.get('sender_user_id', '')) == sender_id:
                breakdown['sender_match'] = WEIGHT_SAME_SENDER
                signals.append("same_sender")

            # Duplicate & Topic Signals
            if msg_text and cand_text and msg_text == cand_text:
                breakdown['exact_duplicate'] = WEIGHT_EXACT_DUPLICATE
                signals.append("exact_duplicate")
            else:
                jaccard = self._compute_jaccard_similarity(msg_words, cand_words)
                if jaccard >= 0.4:
                    breakdown['near_duplicate'] = round(WEIGHT_NEAR_DUPLICATE * jaccard, 2)
                    signals.append(f"near_duplicate(similarity={jaccard:.2f})")

            overlap_words = msg_words.intersection(cand_words)
            if overlap_words:
                overlap_score = len(overlap_words) * WEIGHT_WORD_OVERLAP
                breakdown['keyword_overlap'] = round(overlap_score, 2)
                signals.append(f"keyword_overlap({len(overlap_words)})")

            # Recency Decay Signal
            recency_score, days_ago = self._calculate_recency_score(created_at_str, cand_time_str)
            if recency_score > 0:
                breakdown['recency_decay'] = recency_score

            # Historical Engagement Reaction Signals
            event = self.events_map.get(f"{user_id}_{cand_id}", {})
            if event:
                if event.get('message_reported', 0) in [1, "1"]:
                    breakdown['previous_report'] = WEIGHT_EVENT_REPORTED
                    signals.append("previous_report")
                if event.get('muted_after_message', 0) in [1, "1"]:
                    breakdown['previous_mute'] = WEIGHT_EVENT_MUTED
                    signals.append("previous_mute")
                if event.get('message_replied', 0) in [1, "1"]:
                    breakdown['previous_reply'] = WEIGHT_EVENT_REPLIED
                    signals.append("previous_reply")
                if event.get('message_opened', 0) in [1, "1"]:
                    breakdown['previous_opened'] = WEIGHT_EVENT_OPENED
                    signals.append("previous_opened")

            total_score = round(sum(breakdown.values()), 3)

            # Requires minimum keyword/topic overlap OR exact/near duplicate OR historical engagement
            if overlap_words or 'exact_duplicate' in breakdown or 'near_duplicate' in breakdown or 'previous_report' in breakdown or 'previous_mute' in breakdown:
                if total_score >= 6.0:
                    evidence_results.append(EvidenceDetail(
                        message_id=cand_id,
                        total_score=total_score,
                        score_breakdown=breakdown,
                        triggered_signals=signals
                    ))

        evidence_results.sort(key=lambda x: x.total_score, reverse=True)
        if not evidence_results:
            return []

        top_score = evidence_results[0].total_score
        
        # If top candidate is very strong, return only top candidate unless runner-up is also >= 85% of top score
        filtered_results = [e for e in evidence_results if e.total_score >= top_score * 0.85]

        return filtered_results[:top_k]

    def retrieve_evidence(self, context: Dict[str, Any], top_k: int = 3) -> str:
        """Pipeline backward-compatibility method returning semicolon-separated string or 'none'."""
        details = self.retrieve_evidence_details(context, top_k=top_k)
        if not details:
            return "none"
        return ";".join([d.message_id for d in details])
