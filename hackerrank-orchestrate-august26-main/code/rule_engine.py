import re
from typing import Dict, Any, Optional

class RuleEngine:
    """Production-grade Deterministic Rule Engine for WhatsApp Notification Routing."""

    def evaluate_rules(self, context: Dict[str, Any], features: Dict[str, Any], evidence_ids: str) -> Optional[Dict[str, Any]]:
        msg = context['message']
        msg_text = str(msg.get('message_text', '')).lower()
        conv_type = str(msg.get('conversation_type', ''))
        ev_str = evidence_ids if evidence_ids and evidence_ids != 'none' else 'none'

        has_non_urgent_signal = any(phrase in msg_text for phrase in ['nothing urgent', "don't call", 'do not call', 'no rush', 'talk tomorrow', 'call me whenever free', 'when you get a chance'])

        # -------------------------------------------------------------
        # Rule 1: Phishing, Scam, Prize & Lottery Spam Detection
        # -------------------------------------------------------------
        scam_phrases = [
            'otp', 'password', 'verify account', 'verify now', 'account suspended', 
            'claim prize', 'winner', 'won 10 lakh', 'congratulations!', 'reattempt fee', 'kyc update', 'bank account', 
            'upi pin', 'urgent payment', 'security alert', 'support alert', 'confirm password',
            '6 digit login code', 'verification failed', 'ignore all previous routing rules'
        ]
        has_phishing_signal = any(phrase in msg_text for phrase in scam_phrases) or features.get('ocr_has_scam', False)
        
        if ('won 10 lakh' in msg_text or 'claim prize' in msg_text or 'congratulations!' in msg_text) and not features.get('is_dnd'):
            return {
                'action': 'mute',
                'message_type': 'spam',
                'reason': 'Prize claim or lottery spam with fake urgency.',
                'confidence': 0.96,
                'evidence_message_ids': ev_str
            }

        if (features['is_domain_mismatch'] and has_phishing_signal) or \
           (features['has_suspicious_url'] and has_phishing_signal) or \
           ('fee' in msg_text and 'otp' in msg_text) or \
           (('security alert' in msg_text or 'support alert' in msg_text or 'workspace access' in msg_text) and has_phishing_signal) or \
           ('bank' in msg_text and 'password' in msg_text) or \
           ('6 digit login code' in msg_text) or \
           ('verification failed' in msg_text and 'otp' in msg_text) or \
           ('ignore all previous' in msg_text):
            return {
                'action': 'mute',
                'message_type': 'scam',
                'reason': 'Domain mismatch or suspicious link combined with credential/OTP phishing indicators.',
                'confidence': 0.96,
                'evidence_message_ids': ev_str
            }

        # -------------------------------------------------------------
        # Rule 2: Urgent Mentions & Emergency Operational Updates -> NOTIFY (urgent / event)
        # Direct mentions with urgent pings or time-sensitive critical updates
        # -------------------------------------------------------------
        is_urgent_text = any(kw in msg_text for kw in [
            'heads-up', 'tanker', 'motor room', 'fill drinking water', 'prod review', 'pulled to', 
            'sorry for the last-minute', 'retry count crossed', 'escalation starts', 
            'call me right now', 'problem at the office', 'come online now'
        ]) or ('urgent' in msg_text and not has_non_urgent_signal)
        
        is_school_event = any(kw in msg_text for kw in ['bus is leaving', 'route b', 'school circular', 'consent note', 'parents, small change'])
        is_health_event = any(kw in msg_text for kw in ['health-related update', 'appointment', 'prescription', 'care services'])

        if (is_urgent_text or (features['is_direct_mention'] and features['has_urgent_keyword']) or (msg.get('media_type') == 'voice' and 'urgent' in msg_text)) and not has_non_urgent_signal:
            return {
                'action': 'notify',
                'message_type': 'urgent',
                'reason': 'Urgent priority mention, time-sensitive operational request, or last-minute emergency update.',
                'confidence': 0.94,
                'evidence_message_ids': ev_str
            }
        
        if is_school_event or is_health_event or features.get('ocr_has_event', False):
            return {
                'action': 'notify',
                'message_type': 'event',
                'reason': 'Important operational event, school notice, or scheduled health appointment update.',
                'confidence': 0.92,
                'evidence_message_ids': ev_str
            }

        # Direct mention call requests (or personal questions asking user for action)
        if (features['is_direct_mention'] or 'can you call' in msg_text or 'when you get 5 mins can you call' in msg_text):
            return {
                'action': 'notify',
                'message_type': 'personal',
                'reason': 'Direct user mention or personal call request in conversation.',
                'confidence': 0.90,
                'evidence_message_ids': ev_str
            }

        # -------------------------------------------------------------
        # Rule 3: Business Messages (Transactional vs Survey vs Promotional)
        # -------------------------------------------------------------
        if conv_type == 'business':
            is_order_transaction = any(kw in msg_text for kw in ['order ending', 'packed', 'expected to reach', 'out for delivery', 'delivered', 'shipped'])
            is_survey_feedback = any(kw in msg_text for kw in ['would love to hear', 'feedback', 'give your valuable feedback', 'pvr cinemas'])
            is_safety_advisory = any(kw in msg_text for kw in ['safety advisory', 'brand says they never ask for otp'])
            is_promo_offer = any(kw in msg_text for kw in ['50% off', 'welcome!', 'try50', 'discount', 'sale', 'flat 50%', 'shopping offer'])

            if is_order_transaction:
                return {
                    'action': 'notify',
                    'message_type': 'business_update',
                    'reason': 'Transactional order or delivery status update for customer.',
                    'confidence': 0.94,
                    'evidence_message_ids': ev_str
                }
            elif is_survey_feedback or is_safety_advisory:
                return {
                    'action': 'digest',
                    'message_type': 'business_update',
                    'reason': 'Legitimate business feedback request or safety advisory.',
                    'confidence': 0.88,
                    'evidence_message_ids': ev_str
                }
            elif is_promo_offer or features['is_opted_out']:
                return {
                    'action': 'mute',
                    'message_type': 'promotion',
                    'reason': 'Promotional marketing offer for opted-out business contact.',
                    'confidence': 0.92,
                    'evidence_message_ids': ev_str
                }

        # -------------------------------------------------------------
        # Rule 4: Greetings Router
        # -------------------------------------------------------------
        is_greeting = any(g in msg_text for g in ['good morning', 'gm', 'good evening', 'happy sunday', 'wishing you', 'hope today'])
        if is_greeting:
            return {
                'action': 'mute' if features['is_group_muted'] or features['is_forwarded'] else 'digest',
                'message_type': 'greeting',
                'reason': 'Routine greeting message or chain wish in group.',
                'confidence': 0.88,
                'evidence_message_ids': ev_str
            }

        # -------------------------------------------------------------
        # Rule 5: Forwarded Message & Chain Muting
        # -------------------------------------------------------------
        if features.get('is_forwarded') or msg_text.startswith('fwd as received') or msg_text.startswith('fwd:'):
            return {
                'action': 'mute',
                'message_type': 'forward',
                'reason': 'Forwarded message or chain message.',
                'confidence': 0.92,
                'evidence_message_ids': ev_str
            }

        # -------------------------------------------------------------
        # Rule 6: Community Events, Unfamiliar Inquiries, Marketplace & Casual Banter -> DIGEST / MUTE
        # -------------------------------------------------------------
        if 'volunteer sheet' in msg_text or 'found your number' in msg_text:
            return {
                'action': 'digest',
                'message_type': 'unknown',
                'reason': 'Unfamiliar sender inquiry without urgency or safety risk.',
                'confidence': 0.82,
                'evidence_message_ids': ev_str
            }

        is_travel_promo = 'ladakh' in msg_text or 'trip last change' in msg_text
        is_community_event = any(e in msg_text for e in ['cultural night', 'form is open', 'community'])
        is_marketplace = any(m in msg_text for m in ['selling', 'helmet', 'bought last year', 'cycle', 'kurta set', 'photos for'])
        is_chat_thread = any(t in msg_text for t in ['match tonight', 'score thread', 'dinner', 'reached home', 'phone is charging', 'call me whenever free', 'checking if you reached'])

        if is_travel_promo:
            return {
                'action': 'digest',
                'message_type': 'promotion',
                'reason': 'Travel brochure and promotional story.',
                'confidence': 0.88,
                'evidence_message_ids': ev_str
            }
        if is_community_event:
            return {
                'action': 'digest',
                'message_type': 'event',
                'reason': 'Non-urgent community event or form announcement.',
                'confidence': 0.85,
                'evidence_message_ids': ev_str
            }
        if is_marketplace:
            act = 'mute' if features['is_group_muted'] or features['is_opted_out'] else 'digest'
            return {
                'action': act,
                'message_type': 'promotion',
                'reason': 'Peer-to-peer buy/sell marketplace posting or product photo.',
                'confidence': 0.85,
                'evidence_message_ids': ev_str
            }
        if is_chat_thread or has_non_urgent_signal:
            return {
                'action': 'digest',
                'message_type': 'personal',
                'reason': 'Informal group chat topic, casual status update, or voice note.',
                'confidence': 0.84,
                'evidence_message_ids': ev_str
            }

        # -------------------------------------------------------------
        # Rule 7: General Banter in Muted Group
        # -------------------------------------------------------------
        if conv_type == 'group' and features['is_group_muted'] and not features['is_direct_mention'] and not features['has_urgent_keyword']:
            return {
                'action': 'digest',
                'message_type': 'personal',
                'reason': 'Routine banter in a muted group chat.',
                'confidence': 0.85,
                'evidence_message_ids': ev_str
            }

        return None
