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
WEIGHT_BM25 = 3.5
WEIGHT_FUZZY_RATIO = 8.0

WEIGHT_EVENT_REPORTED = 10.0
WEIGHT_EVENT_MUTED = 4.0
WEIGHT_EVENT_REPLIED = 3.0
WEIGHT_EVENT_OPENED = 1.0

WEIGHT_EXACT_DUPLICATE = 15.0

RECENCY_DECAY_HALF_LIFE_DAYS = 90.0
MAX_RECENCY_BONUS = 2.0

BM25_K1 = 1.5
BM25_B = 0.75

STOP_WORDS = {'the', 'a', 'an', 'and', 'or', 'is', 'in', 'at', 'of', 'to', 'for', 'with', 'on', 'this', 'that', 'it', 'you', 'your', 'i', 'we', 'my', 'be', 'are', 'have', 'has', 'pls', 'please'}

@dataclass
class EvidenceDetail:
    message_id: str
    total_score: float
    score_breakdown: Dict[str, float]
    triggered_signals: List[str]

class EvidenceRetriever:
    """Research-Grade BM25 + Fast Fuzzy Token Evidence Retrieval Engine."""

    def __init__(self, data_loader: DataLoader):
        self.dl = data_loader
        self.events_map: Dict[str, Dict[str, Any]] = {}
        self.idf_map: Dict[str, float] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.doc_token_counts: Dict[str, Dict[str, int]] = {}
        self.avg_doc_len: float = 1.0
        
        if hasattr(self.dl, 'message_events_df') and self.dl.message_events_df is not None:
            for _, row in self.dl.message_events_df.iterrows():
                m_id = str(row['message_id'])
                u_id = str(row['user_id'])
                self.events_map[f"{u_id}_{m_id}"] = row.to_dict()

        self._build_bm25_corpus()

    def _build_bm25_corpus(self):
        """Computes Inverse Document Frequency (IDF) and BM25 document statistics across history corpus."""
        msg_hist = getattr(self.dl, 'message_history_df', None)
        if msg_hist is None:
            msg_hist = getattr(self.dl, 'message_history', [])

        corpus = msg_hist if isinstance(msg_hist, list) else [r.to_dict() for _, r in msg_hist.iterrows()]
        total_docs = len(corpus)

        if total_docs == 0:
            return

        doc_freq: Dict[str, int] = {}
        total_words = 0

        for item in corpus:
            cand_id = str(getattr(item, 'message_id', item.get('message_id', '') if isinstance(item, dict) else ''))
            text = str(getattr(item, 'message_text', item.get('message_text', '') if isinstance(item, dict) else '')).lower()
            words = [w for w in re.findall(r'\w+', text) if len(w) > 2 and w not in STOP_WORDS]
            
            self.doc_lengths[cand_id] = len(words)
            total_words += len(words)

            counts: Dict[str, int] = {}
            for w in words:
                counts[w] = counts.get(w, 0) + 1
            self.doc_token_counts[cand_id] = counts

            for w in counts.keys():
                doc_freq[w] = doc_freq.get(w, 0) + 1

        self.avg_doc_len = total_words / total_docs if total_docs > 0 else 1.0

        for word, freq in doc_freq.items():
            self.idf_map[word] = math.log((total_docs - freq + 0.5) / (freq + 0.5) + 1.0)

    def _compute_bm25_score(self, cand_id: str, query_words: List[str]) -> float:
        """Computes Okapi BM25 relevance score using pre-indexed token counts."""
        if not query_words:
            return 0.0
        
        doc_len = self.doc_lengths.get(cand_id, 0)
        cand_counts = self.doc_token_counts.get(cand_id, {})
        if not cand_counts:
            return 0.0

        score = 0.0

        for word in query_words:
            if word not in cand_counts:
                continue
            tf = cand_counts[word]
            idf = self.idf_map.get(word, 1.0)
            
            num = tf * (BM25_K1 + 1.0)
            den = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * (doc_len / self.avg_doc_len))
            score += idf * (num / den)

        return round(score, 3)

    def _compute_fuzzy_token_ratio(self, s1: str, s2: str) -> float:
        if not s1 or not s2:
            return 0.0
        len1, len2 = len(s1), len(s2)
        if min(len1, len2) / max(len1, len2) < 0.40:
            return 0.0
        return SequenceMatcher(None, s1, s2).ratio()

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
        
        msg_words = [w for w in re.findall(r'\w+', msg_text) if len(w) > 2 and w not in STOP_WORDS]

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

            # Exact Match & Fuzzy Sequence Signals
            if msg_text and cand_text and msg_text == cand_text:
                breakdown['exact_duplicate'] = WEIGHT_EXACT_DUPLICATE
                signals.append("exact_duplicate")
            else:
                fuzzy_sim = self._compute_fuzzy_token_ratio(msg_text, cand_text)
                if fuzzy_sim >= 0.40:
                    breakdown['fuzzy_ratio'] = round(WEIGHT_FUZZY_RATIO * fuzzy_sim, 2)
                    signals.append(f"fuzzy_ratio({fuzzy_sim:.2f})")

            # Okapi BM25 Relevance Score
            bm25 = self._compute_bm25_score(cand_id, msg_words)
            if bm25 > 0.0:
                breakdown['bm25_relevance'] = round(WEIGHT_BM25 * bm25, 2)
                signals.append(f"bm25_relevance({bm25:.2f})")

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

            if bm25 > 0.10 or 'exact_duplicate' in breakdown or 'fuzzy_ratio' in breakdown or 'previous_report' in breakdown or 'previous_mute' in breakdown:
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
