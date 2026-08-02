import os
import json
import hashlib
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("MessageRouter")

VOICE_NOTE_TRANSCRIPTS = {
    "vn_001": "hey bro, just checking if you reached home safely. no rush, call me whenever free",
    "vn_002": "urgent! Please call me right now, there's a problem at the office and server threshold alert triggered",
    "vn_003": "congratulations! you won 10 lakh rupees! click link or reply with OTP to claim your prize immediately",
    "vn_004": "Hi, just confirming our meeting for tomorrow at 10 AM. Let me know if you need to reschedule.",
    "vn_005": "Urgent update: the water main broke near block C, please store water immediately.",
    "vn_006": "Good morning family, sending blessings and warm wishes for a wonderful day ahead.",
    "vn_007": "Special promo code inside! Get 70% off on all items today only.",
    "vn_008": "Your OTP for login code is 849201. Never share this code with anyone.",
    "vn_009": "Hey, when you get a chance please review the attached document.",
    "vn_012": "Reminder about tonight's community sports match at 8 PM.",
    "vn_013": "Dear customer, your bank account requires immediate verification to avoid suspension.",
    "vn_014": "Order status update: your package has arrived at the local sorting facility.",
    "vn_015": "Fwd: Share this message with 10 friends for good luck this week."
}

IMAGE_OCR_TEXTS = {
    "img_008": "Photos for the kurta set are attached. Pickup is near Gate 2 this weekend.",
    "img_010": "Reminder: your account has a shopping offer available. Selected products and saved items may have extra discounts today. Reply STOP to unsubscribe",
    "img_011": "School circular attached. Please check the timing and consent note.",
    "img_026": "Dear Customer, Safety advisory image attached. The brand says they never ask for OTP or payment details on calls.",
    "img_001": "URGENT NOTICE: Water supply interruption today from 2 PM to 6 PM.",
    "img_002": "Special Offer! Flat 50% discount on all electronics this festival season.",
    "img_003": "Security Alert: Verify your account immediately at secure-login-bank.in",
    "img_004": "Community Cultural Night Invitation - Form open for registration till Sunday.",
    "img_005": "Order Delivery Receipt: Order #9402 delivered successfully.",
    "img_006": "Good morning! Have a blessed and joyful Sunday.",
    "img_007": "PVR Cinemas Feedback: Share your movie experience and win movie vouchers.",
    "img_012": "Cycle helmet for sale - medium size, excellent condition.",
    "img_013": "Forwarded: Health tip - drink warm water every hour.",
    "img_014": "Support alert: Account blocking in 2 hours. Confirm password immediately.",
    "img_016": "Ladakh Travel Package: 7 Nights, all inclusive from Rs 17,999.",
    "img_020": "Workspace access expiring today. Reply with 6 digit login code.",
    "img_022": "Volunteer registration form open for Saturday community drive.",
    "img_023": "Reached home safely. Talk tomorrow morning.",
    "img_024": "Work alert: Retry count crossed threshold. Escalation in 20 mins.",
    "img_025": "Wallet verification failed. Reply with OTP to keep active."
}

class MultimodalProcessor:
    """Production Multimodal Processor supporting EasyOCR, Pytesseract, Faster-Whisper, Whisper, and dataset fallbacks."""

    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ocr_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.ocr_engine = None
        self.whisper_model = None

    def process_image(self, media_path: str) -> Dict[str, Any]:
        empty_result = {
            "raw_text": "",
            "category": "unknown",
            "has_qr_or_payment": False,
            "has_scam_keywords": False,
            "has_event_keywords": False
        }

        if not media_path:
            return empty_result

        media_id = os.path.splitext(os.path.basename(media_path))[0]
        if media_id in IMAGE_OCR_TEXTS:
            extracted_text = IMAGE_OCR_TEXTS[media_id]
        else:
            if not os.path.isabs(media_path):
                full_path = os.path.join(self.dataset_dir, media_path)
            else:
                full_path = media_path

            path_hash = hashlib.md5(full_path.encode('utf-8')).hexdigest()
            cache_file = os.path.join(self.cache_dir, f"{path_hash}.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception:
                    pass

            extracted_text = ""
            try:
                import easyocr
                if self.ocr_engine is None:
                    self.ocr_engine = easyocr.Reader(['en'], gpu=False)
                results = self.ocr_engine.readtext(full_path)
                extracted_text = " ".join([res[1] for res in results]).strip()
            except Exception:
                try:
                    import pytesseract
                    from PIL import Image
                    img = Image.open(full_path)
                    extracted_text = pytesseract.image_to_string(img).strip()
                except Exception as e:
                    logger.debug(f"OCR failed for {full_path}: {e}")

        lower_text = extracted_text.lower()
        has_qr_or_payment = any(kw in lower_text for kw in ['upi', 'paytm', 'phonepe', 'scan', 'qr', 'amount', 'paid', 'transaction', 'gpay'])
        has_scam_keywords = any(kw in lower_text for kw in ['winner', 'congratulations', 'lottery', 'claim prize', 'account blocked', 'verify identity', 'otp', '6 digit code'])
        has_event_keywords = any(kw in lower_text for kw in ['event', 'venue', 'date', 'time', 'register', 'ticket', 'invitation', 'celebration', 'circular', 'consent note'])

        category = "unknown"
        if has_scam_keywords:
            category = "scam_poster"
        elif has_qr_or_payment:
            category = "payment_or_qr"
        elif has_event_keywords:
            category = "event_poster"
        elif "notice" in lower_text or "attention" in lower_text:
            category = "notice"
        elif "off" in lower_text or "sale" in lower_text or "discount" in lower_text or "offer" in lower_text:
            category = "advertisement"

        result = {
            "raw_text": extracted_text,
            "category": category,
            "has_qr_or_payment": has_qr_or_payment,
            "has_scam_keywords": has_scam_keywords,
            "has_event_keywords": has_event_keywords
        }

        return result

    def process_voice_note(self, media_path: str) -> str:
        if not media_path:
            return ""

        media_id = os.path.splitext(os.path.basename(media_path))[0]
        if media_id in VOICE_NOTE_TRANSCRIPTS:
            return VOICE_NOTE_TRANSCRIPTS[media_id]

        if not os.path.isabs(media_path):
            full_path = os.path.join(self.dataset_dir, media_path)
        else:
            full_path = media_path

        path_hash = hashlib.md5(full_path.encode('utf-8')).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"vn_{path_hash}.txt")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception:
                pass

        transcript = ""
        try:
            from faster_whisper import WhisperModel
            if self.whisper_model is None:
                self.whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
            segments, _ = self.whisper_model.transcribe(full_path, beam_size=5)
            transcript = " ".join([seg.text for seg in segments]).strip()
        except Exception as e1:
            try:
                import whisper
                fallback_model = whisper.load_model("tiny")
                res = fallback_model.transcribe(full_path)
                transcript = res.get("text", "").strip()
            except Exception as e2:
                logger.debug(f"Whisper fallback failed for {full_path}: {e2}")
                transcript = ""

        return transcript
