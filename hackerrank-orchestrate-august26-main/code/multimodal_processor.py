import os
import logging
from typing import Optional, Dict

logger = logging.getLogger("MessageRouter")

class MultimodalProcessor:
    """OCR (Pytesseract/EasyOCR fallback) and ASR (Faster-Whisper/Whisper fallback) with disk caching."""

    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.cache: Dict[str, str] = {}
        self.ocr_engine = None
        self.whisper_model = None

    def process_image(self, media_path: str) -> str:
        """Extracts text from image posters/screenshots with disk caching."""
        if not media_path:
            return ""

        full_path = os.path.join(self.dataset_dir, media_path) if not os.path.isabs(media_path) else media_path
        if not os.path.exists(full_path):
            return ""

        if full_path in self.cache:
            return self.cache[full_path]

        extracted_text = ""
        # Try pytesseract / easyocr
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(full_path)
            extracted_text = pytesseract.image_to_string(img).strip()
        except Exception:
            try:
                import easyocr
                if self.ocr_engine is None:
                    self.ocr_engine = easyocr.Reader(['en'])
                results = self.ocr_engine.readtext(full_path)
                extracted_text = " ".join([res[1] for res in results]).strip()
            except Exception as e:
                logger.debug(f"OCR not available or failed for {full_path}: {e}")

        self.cache[full_path] = extracted_text
        return extracted_text

    def process_voice_note(self, media_path: str) -> str:
        """Transcribes voice notes into text using Faster-Whisper/Whisper with caching."""
        if not media_path:
            return ""

        full_path = os.path.join(self.dataset_dir, media_path) if not os.path.isabs(media_path) else media_path
        if not os.path.exists(full_path):
            return ""

        if full_path in self.cache:
            return self.cache[full_path]

        transcript = ""
        try:
            from faster_whisper import WhisperModel
            if self.whisper_model is None:
                self.whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
            segments, _ = self.whisper_model.transcribe(full_path)
            transcript = " ".join([seg.text for seg in segments]).strip()
        except Exception:
            try:
                import whisper
                if self.whisper_model is None:
                    self.whisper_model = whisper.load_model("tiny")
                res = self.whisper_model.transcribe(full_path)
                transcript = res.get("text", "").strip()
            except Exception as e:
                logger.debug(f"Whisper ASR not available or failed for {full_path}: {e}")

        self.cache[full_path] = transcript
        return transcript
