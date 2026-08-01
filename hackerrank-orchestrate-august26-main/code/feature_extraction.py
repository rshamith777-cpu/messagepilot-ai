import re
from datetime import datetime
from typing import Dict, Any

class FeatureExtractor:
    """Extracts explicit behavioral and domain features from enriched context."""

    def extract_features(self, context: Dict[str, Any]) -> Dict[str, Any]:
        msg = context['message']
        user = context['user']
        group = context['group']
        group_member = context['group_member']
        sender_member = context['sender_group_member']
        business = context['business']
        user_business = context['user_business']

        msg_text = str(msg.get('message_text', ''))
        user_id = str(msg.get('user_id', ''))
        
        # 1. DND Window Feature
        created_at_str = str(msg.get('created_at', ''))
        is_dnd = False
        dnd_window = str(user.get('do_not_disturb_window', ''))
        if dnd_window and '-' in dnd_window and created_at_str:
            try:
                msg_time = datetime.strptime(created_at_str, "%Y-%m-%d %H:%M").time()
                start_str, end_str = dnd_window.split('-')
                start_time = datetime.strptime(start_str.strip(), "%H:%M").time()
                end_time = datetime.strptime(end_str.strip(), "%H:%M").time()
                
                if start_time <= end_time:
                    is_dnd = start_time <= msg_time <= end_time
                else: # Overnight window e.g. 22:00 - 07:00
                    is_dnd = msg_time >= start_time or msg_time <= end_time
            except Exception:
                pass

        # 2. Business & Domain Mismatch Features
        official_domain = str(business.get('official_domain', '')).lower()
        domain_used = str(business.get('domain_used_by_sender', '')).lower()
        is_domain_mismatch = bool(official_domain and domain_used and official_domain != domain_used)
        
        # Check text for suspicious external links
        urls = re.findall(r'https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9-]+\.[a-zA-Z]{2,}', msg_text)
        has_suspicious_url = False
        if official_domain and urls:
            for url in urls:
                if official_domain not in url.lower() and ('http' in url.lower() or '.in' in url.lower() or '.com' in url.lower() or 'pay' in url.lower()):
                    has_suspicious_url = True

        # 3. Business Opt-Out Feature
        allows_promotions = user_business.get('allows_promotions', 1)
        promotions_opted_out_at = str(user_business.get('promotions_opted_out_at', ''))
        is_opted_out = (allows_promotions == 0 or allows_promotions == "0" or bool(promotions_opted_out_at))

        # 4. Group Member & Admin Features
        is_group_muted = bool(group_member.get('group_muted_by_user', 0) in [1, "1", True])
        sender_is_admin = str(sender_member.get('role', '')).lower() == 'admin'
        
        # 5. Direct Mention & Urgency Keywords
        is_direct_mention = f"@{user_id}" in msg_text or "@all" in msg_text or "@channel" in msg_text
        urgent_keywords = ['urgent', 'emergency', 'asap', 'immediately', 'heads-up', 'action required', 'attention', 'deadline', 'early', 'cancelled', 'pls fill', 'missing']
        has_urgent_keyword = any(kw in msg_text.lower() for kw in urgent_keywords)

        # 6. Forwarded Message Indicators
        fwd_count = int(msg.get('forwarded_count', 0) or 0)
        fwd_text_prefixes = ['fwd:', 'forwarded:', 'forwarded as received', 'fwd', 'forwarded']
        has_fwd_prefix = any(msg_text.strip().lower().startswith(prefix) for prefix in fwd_text_prefixes)
        is_forwarded = fwd_count > 0 or has_fwd_prefix

        return {
            'is_dnd': is_dnd,
            'is_domain_mismatch': is_domain_mismatch,
            'has_suspicious_url': has_suspicious_url,
            'is_opted_out': is_opted_out,
            'is_group_muted': is_group_muted,
            'sender_is_admin': sender_is_admin,
            'is_direct_mention': is_direct_mention,
            'has_urgent_keyword': has_urgent_keyword,
            'forwarded_count': fwd_count,
            'is_forwarded': is_forwarded
        }
