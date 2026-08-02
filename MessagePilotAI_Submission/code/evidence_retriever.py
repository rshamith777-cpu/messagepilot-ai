import re
import math
from difflib import SequenceMatcher
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from data_loader import DataLoader

WEIGHT_BUSINESS_MATCH = 4.0
WEIGHT_GROUP_MATCH = 3.0
WEIGHT_SAME_SENDER = 2.0
WEIGHT_TFIDF_COSINE = 6.0
WEIGHT_FUZZY_RATIO = 8.0

WEIGHT_EVENT_REPORTED = 10.0
WEIGHT_EVENT_MUTED = 4.0
WEIGHT_EVENT_REPLIED = 3.0
WEIGHT_EVENT_OPENED = 1.0

WEIGHT_EXACT_DUPLICATE = 15.0

RECENCY_DECAY_HALF_LIFE_DAYS = 90.0
MAX_RECENCY_BONUS = 2.0

@dataclass
class EvidenceDetail:
    message_id: str
    total_score: float
    score_breakdown: Dict[str, float]
    triggered_signals: List[str]

class EvidenceRetriever:
    """Production Evidence Retrieval Engine with TF-IDF Weighting, Fuzzy Ratio Clustering, and Additive Scoring."""

    def __init__(self, data_loader: DataLoader):
        self.dl = data_loader
        self.events_map: Dict[str, Dict[str, Any]] = {}
        self.idf_map: Dict[str, float] = {}
        
        if hasattr(self.dl, 'message_events_df') and self.dl.message_events_df is not None:
            for _, row in self.dl.message_events_df.iterrows():
                m_id = str(row['message_id'])
                u_id = str(row['user_id'])
                self.events_map[f"{u_id}_{m_id}"] = row.to_dict()

        self._build_idf_corpus()

    def _build_idf_corpus(self):
        """Computes Inverse Document Frequency (IDF) weights across message history corpus."""
        msg_hist = getattr(self.dl, 'message_history_df', None)
        if msg_hist is None:
            msg_hist = getattr(self.dl, 'message_history', [])

        total_docs = 0
        doc_freq: Dict[str, int] = {}
        stop_words = {'the', 'a', 'an', 'and', 'or', 'is', 'in', 'at', 'of', 'to', 'for', 'with', 'on', 'this', 'that', 'it', 'you', 'your', 'i', 'we', 'my', 'be', 'are', 'have', 'has', 'pls', 'please'}

        corpus = msg_hist if isinstance(msg_hist, list) else [r.to_dict() for _, r in msg_hist.iterrows()]
        total_docs = len(corpus)

        for item in corpus:
            text = str(getattr(item, 'message_text', item.get('message_text', '') if isinstance(item, dict) else '')).lower()
            words = set([w for w in re.findall(r'\w+', text) if len(w) > 2 and w not in stop_words])
            for w in words:
                doc_freq[w] = doc_freq.get(w, 0) + 1

        for word, freq in doc_freq.items():
            self.idf_map[word] = math.log((total_docs + 1.0) / (freq + 1.0)) + 1.0

    def _compute_fuzzy_ratio(self, s1: str, s2: str) -> float:
        if not s1 or not s2:
            return 0.0
        return SequenceMatcher(None, s1, s2).ratio()

    def _compute_tfidf_cosine(self, words1: set, words2: set) -> float:
        if not words1 or not words2:
            return 0.0
        shared = words1.intersection(words2)
        if not shared:
            return 0.0
        
        shared_weight = sum(self.idf_map.get(w, 1.0) for w in shared)
        total_weight1 = math.sqrt(sum(self.idf_map.get(w, 1.0) ** 2 for w in words1))
        total_weight2 = math.sqrt(sum(self.idf_map.get(w, 1.0) ** 2 for w in words2))

        return shared_weight / (total_weight1 * total_weight2) if (total_weight1 * total_weight2) > 0 else 0.0

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

            # Duplicate & Fuzzy Similarity Signals
            if msg_text and cand_text and msg_text == cand_text:
                breakdown['exact_duplicate'] = WEIGHT_EXACT_DUPLICATE
                signals.append("exact_duplicate")
            else:
                fuzzy_sim = self._compute_fuzzy_ratio(msg_text, cand_text)
                if fuzzy_sim >= 0.40:
                    breakdown['fuzzy_ratio'] = round(WEIGHT_FUZZY_RATIO * fuzzy_sim, 2)
                    signals.append(f"fuzzy_ratio({fuzzy_sim:.2f})")

            # TF-IDF Cosine Similarity Signal
            tfidf_sim = self._compute_tfidf_cosine(msg_words, cand_words)
            if tfidf_sim > 0.0:
                breakdown['tfidf_cosine'] = round(WEIGHT_TFIDF_COSINE * tfidf_sim, 2)
                signals.append(f"tfidf_cosine({tfidf_sim:.2f})")

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

            if tfidf_sim > 0.10 or 'exact_duplicate' in breakdown or 'fuzzy_ratio' in breakdown or 'previous_report' in breakdown or 'previous_mute' in breakdown:
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
        filtered_results = [e for e in evidence_results if e.total_score >= top_score * 0.85]

        return filtered_results[:top_k]

    def retrieve_evidence(self, context: Dict[str, Any], top_k: int = 3) -> str:
        """Pipeline backward-compatibility method returning semicolon-separated string or 'none'."""
        details = self.retrieve_evidence_details(context, top_k=top_k)
        if not details:
            return "none"
        return ";".join([d.message_id for d in details])
