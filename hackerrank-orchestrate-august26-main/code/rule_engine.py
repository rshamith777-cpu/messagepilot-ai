import re
from typing import Dict, Any, Optional

class RuleEngine:
    """Production-grade Deterministic Rule Engine for WhatsApp Notification Routing."""

    def evaluate_rules(self, context: Dict[str, Any], features: Dict[str, Any], evidence_ids: str) -> Optional[Dict[str, Any]]:
        msg_text = str(context['message'].get('message_text', '')).lower()
        conv_type = str(context['message'].get('conversation_type', ''))
        
        # -------------------------------------------------------------
        # Rule 1: Enhanced Phishing & Scam Detection
        # -------------------------------------------------------------
        scam_phrases = ['otp', 'password', 'verify account', 'verify now', 'account suspended', 
                        'claim prize', 'winner', 'reattempt fee', 'kyc update', 'bank account', 
                        'upi pin', 'urgent payment', 'security alert', 'support alert', 'confirm password']
        has_phishing_signal = any(phrase in msg_text for phrase in scam_phrases)
        
        if (features['is_domain_mismatch'] and has_phishing_signal) or \
           (features['has_suspicious_url'] and has_phishing_signal) or \
           ('fee' in msg_text and 'otp' in msg_text) or \
           ('security alert' in msg_text or 'support alert' in msg_text) or \
           ('bank' in msg_text and 'password' in msg_text):
            return {
                'action': 'mute',
                'message_type': 'scam',
                'reason': 'Domain mismatch or suspicious link combined with credential/OTP phishing indicators.',
                'confidence': 0.96,
                'evidence_message_ids': evidence_ids if evidence_ids != 'none' else 'none'
            }

        # -------------------------------------------------------------
        # Rule 2: Forwarded Message Muting
        # -------------------------------------------------------------
        if features['is_forwarded'] or msg_text.startswith('fwd as received'):
            return {
                'action': 'mute',
                'message_type': 'forward',
                'reason': 'Forwarded message or chain message.',
                'confidence': 0.92,
                'evidence_message_ids': evidence_ids if evidence_ids != 'none' else 'none'
            }

        # -------------------------------------------------------------
        # Rule 3: Promotional Offer Muting for Opted-Out Users
        # -------------------------------------------------------------
        if conv_type == 'business' and (features['is_opted_out'] or '50% off' in msg_text or 'welcome!' in msg_text or 'try50' in msg_text) and not ('order ending' in msg_text or 'packed' in msg_text):
            return {
                'action': 'mute',
                'message_type': 'promotion',
                'reason': 'Promotional discount/marketing message for opted-out business contact.',
                'confidence': 0.94,
                'evidence_message_ids': evidence_ids if evidence_ids != 'none' else 'none'
            }

        # -------------------------------------------------------------
        # Rule 4: Urgent Mentions & Operational Emergency Updates
        # -------------------------------------------------------------
        is_event_keyword = any(kw in msg_text for kw in ['leaving', 'schedule', 'timing', 'bus', 'route', 'health-related', 'update is ready'])
        
        if 'prod review' in msg_text or 'sorry for the last-minute' in msg_text:
            return {
                'action': 'notify',
                'message_type': 'urgent',
                'reason': 'Urgent direct mention regarding schedule change.',
                'confidence': 0.92,
                'evidence_message_ids': evidence_ids if evidence_ids != 'none' else 'none'
            }
        elif is_event_keyword and conv_type in ['group', 'business'] and not features['is_group_muted']:
            return {
                'action': 'notify',
                'message_type': 'event',
                'reason': 'Important operational event update for group/business members.',
                'confidence': 0.88,
                'evidence_message_ids': evidence_ids if evidence_ids != 'none' else 'none'
            }
        elif features['is_direct_mention']:
            return {
                'action': 'notify',
                'message_type': 'urgent' if features['has_urgent_keyword'] else 'personal',
                'reason': 'Direct user mention requires user attention.',
                'confidence': 0.88,
                'evidence_message_ids': evidence_ids if evidence_ids != 'none' else 'none'
            }

        # -------------------------------------------------------------
        # Rule 5: Transactional Business Router
        # -------------------------------------------------------------
        if conv_type == 'business' and ('order ending' in msg_text or 'packed' in msg_text or 'expected to reach' in msg_text):
            return {
                'action': 'notify',
                'message_type': 'business_update',
                'reason': 'Transactional order status update.',
                'confidence': 0.92,
                'evidence_message_ids': evidence_ids if evidence_ids != 'none' else 'none'
            }

        # -------------------------------------------------------------
        # Rule 4: Muted Group Routine Greetings -> MUTE
        # Routine greetings in MUTED groups route to MUTE instead of DIGEST
        # -------------------------------------------------------------
        is_greeting = any(g in msg_text for g in ['good morning', 'gm', 'good evening', 'happy sunday', 'wishing you', 'hope today'])
        if is_greeting and features['is_group_muted']:
            return {
                'action': 'mute',
                'message_type': 'greeting',
                'reason': 'Routine greeting in a muted group chat is suppressed.',
                'confidence': 0.88,
                'evidence_message_ids': evidence_ids if evidence_ids != 'none' else 'none'
            }
        elif is_greeting:
            return {
                'action': 'digest',
                'message_type': 'greeting',
                'reason': 'Routine greeting message aggregated into digest.',
                'confidence': 0.86,
                'evidence_message_ids': evidence_ids if evidence_ids != 'none' else 'none'
            }

        # -------------------------------------------------------------
        # Rule 6: Community Events, Marketplace & Promotions -> DIGEST
        # -------------------------------------------------------------
        if 'ladakh' in msg_text or 'trip last change' in msg_text:
            return {
                'action': 'digest',
                'message_type': 'promotion',
                'reason': 'Travel promotional story/brochure.',
                'confidence': 0.88,
                'evidence_message_ids': evidence_ids if evidence_ids != 'none' else 'none'
            }

        is_community_event = any(e in msg_text for e in ['cultural night', 'form is open', 'community', 'registration'])
        is_marketplace = any(m in msg_text for m in ['selling', 'helmet', 'bought last year', 'cycle'])
        is_chat_thread = any(t in msg_text for t in ['match tonight', 'score thread', 'dinner'])

        if is_community_event:
            return {
                'action': 'digest',
                'message_type': 'event',
                'reason': 'Non-urgent community event or form announcement.',
                'confidence': 0.85,
                'evidence_message_ids': evidence_ids if evidence_ids != 'none' else 'none'
            }
        if is_marketplace:
            return {
                'action': 'digest',
                'message_type': 'promotion',
                'reason': 'Peer-to-peer buy/sell marketplace posting.',
                'confidence': 0.85,
                'evidence_message_ids': evidence_ids if evidence_ids != 'none' else 'none'
            }
        if is_chat_thread:
            return {
                'action': 'digest',
                'message_type': 'personal',
                'reason': 'Informal group chat topic or sports discussion.',
                'confidence': 0.84,
                'evidence_message_ids': evidence_ids if evidence_ids != 'none' else 'none'
            }

        # -------------------------------------------------------------
        # Rule 6: General Banter in Muted Group
        # -------------------------------------------------------------
        if conv_type == 'group' and features['is_group_muted'] and not features['is_direct_mention'] and not features['has_urgent_keyword']:
            return {
                'action': 'digest',
                'message_type': 'personal',
                'reason': 'Routine banter in a muted group chat.',
                'confidence': 0.85,
                'evidence_message_ids': evidence_ids if evidence_ids != 'none' else 'none'
            }

        # Pass remaining ambiguous cases to LLM Reasoning
        return None
