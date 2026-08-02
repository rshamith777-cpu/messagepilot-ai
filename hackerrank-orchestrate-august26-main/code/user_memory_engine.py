from typing import Dict, Any, List

class UserMemoryEngine:
    """Production User Memory Engine tracking historical user engagement signals (open rate, reply rate, mute rate, report rate)."""

    def __init__(self, data_loader: Any):
        self.dl = data_loader
        self.user_stats: Dict[str, Dict[str, float]] = {}
        self._build_user_memory()

    def _build_user_memory(self):
        """Aggregates user interaction stats across message history and events."""
        events_df = getattr(self.dl, 'message_events_df', None)
        if events_df is None or not hasattr(events_df, 'iterrows'):
            return

        user_counts: Dict[str, Dict[str, int]] = {}

        for _, row in events_df.iterrows():
            u_id = str(row['user_id'])
            if u_id not in user_counts:
                user_counts[u_id] = {'total': 0, 'opened': 0, 'replied': 0, 'muted': 0, 'reported': 0}
            
            user_counts[u_id]['total'] += 1
            if row.get('message_opened', 0) in [1, '1']:
                user_counts[u_id]['opened'] += 1
            if row.get('message_replied', 0) in [1, '1']:
                user_counts[u_id]['replied'] += 1
            if row.get('muted_after_message', 0) in [1, '1']:
                user_counts[u_id]['muted'] += 1
            if row.get('message_reported', 0) in [1, '1']:
                user_counts[u_id]['reported'] += 1

        for u_id, cnt in user_counts.items():
            tot = max(1, cnt['total'])
            self.user_stats[u_id] = {
                'open_rate': round(cnt['opened'] / tot, 3),
                'reply_rate': round(cnt['replied'] / tot, 3),
                'mute_rate': round(cnt['muted'] / tot, 3),
                'report_rate': round(cnt['reported'] / tot, 3)
            }

    def get_user_personalization_modifier(self, user_id: str) -> Dict[str, float]:
        """Returns personalized score modifiers for Notify, Digest, and Mute based on historical user traits."""
        stats = self.user_stats.get(str(user_id), {'open_rate': 0.5, 'reply_rate': 0.2, 'mute_rate': 0.1, 'report_rate': 0.0})
        
        notify_mod = 0.0
        digest_mod = 0.0
        mute_mod = 0.0

        if stats['reply_rate'] > 0.4:
            notify_mod += 5.0
        if stats['mute_rate'] > 0.3:
            mute_mod += 10.0
        if stats['report_rate'] > 0.1:
            mute_mod += 15.0

        return {
            'notify_modifier': notify_mod,
            'digest_modifier': digest_mod,
            'mute_modifier': mute_mod,
            'user_stats': stats
        }
