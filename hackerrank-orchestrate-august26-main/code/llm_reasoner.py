import os
import json
import time
import logging
import urllib.request
import urllib.parse
from typing import Dict, Any, List, Optional
from models import MessageContext, logger
from evidence_retriever import EvidenceDetail

# Prompt template enforcing the 10 step reasoning process and strict JSON output schema
SYSTEM_PROMPT = """You are an AI WhatsApp Message Notification Router.
Your job is to decide whether an incoming message should be:
- "notify": interrupt the user now (urgent, time-sensitive, important direct mentions, security alerts)
- "digest": safe but non-urgent content to show later
- "mute": low-value, repetitive, unwanted, suspicious, scam, or opted-out content

Reason step-by-step:
1. Understand the message content.
2. Determine the best message_type out of: ['personal', 'urgent', 'event', 'payment', 'business_update', 'promotion', 'greeting', 'forward', 'spam', 'scam', 'unknown']
3. Assess urgency.
4. Assess scam/phishing risk.
5. Review provided evidence messages.
6. Consider recipient personalization (DND window, opt-out status, muted groups).
7. Compare: notify vs digest vs mute.
8. Select the best action.
9. Write ONE short, concise human-readable sentence as the reason.
10. Return strictly valid JSON.

Input Context:
- Current Message: {current_message}
- Extracted Features: {extracted_features}
- Baseline Rule Recommendation: {baseline_recommendation}
- Selected Historical Evidence: {selected_evidence}
- Personalization Context: {personalization_context}

Output strictly valid JSON with this exact schema (no markdown, no backticks, no explanatory text):
{{
  "action": "notify" | "digest" | "mute",
  "message_type": "personal" | "urgent" | "event" | "payment" | "business_update" | "promotion" | "greeting" | "forward" | "spam" | "scam" | "unknown",
  "reason": "one concise sentence explaining decision",
  "confidence": 0.85,
  "evidence_message_ids": ["msg_id1", ...]
}}
"""

import hashlib

class LLMProvider:
    """Abstract Base Class for Provider API Abstraction."""
    def generate(self, prompt: str, timeout: int = 15, max_retries: int = 4) -> str:
        raise NotImplementedError

    def _execute_http_with_retry(self, req: urllib.request.Request, timeout: int = 15, max_retries: int = 4) -> str:
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                # Clone request object for retries
                curr_req = urllib.request.Request(req.full_url, data=req.data, headers=req.headers)
                with urllib.request.urlopen(curr_req, timeout=timeout) as resp:
                    return resp.read().decode('utf-8')
            except urllib.error.HTTPError as e:
                last_error = e
                if e.code == 429 and attempt < max_retries:
                    backoff = 3.0 * (2 ** attempt)  # 3s, 6s, 12s, 24s backoff for 429
                    print(f"  [429 Rate Limit] Retrying in {backoff:.1f}s (Attempt {attempt+1}/{max_retries})...")
                    time.sleep(backoff)
                elif attempt < max_retries:
                    time.sleep(1.0)
                else:
                    raise last_error
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    time.sleep(1.0)
                else:
                    raise last_error
        raise last_error

class OpenRouterProvider(LLMProvider):
    def __init__(self, api_key: str, model_name: Optional[str] = None):
        self.api_key = api_key
        self.model = model_name or os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash")
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def generate(self, prompt: str, timeout: int = 15, max_retries: int = 2) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.1
        }
        masked_key = self.api_key[:8] + "..." + self.api_key[-4:] if len(self.api_key) > 12 else "***"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'HTTP-Referer': 'https://hackerrank.com',
            'X-Title': 'WhatsApp Notification Router'
        }
        
        print("\n================ OPENROUTER HTTP DEBUG ================")
        print(f"1. Endpoint URL: {self.url}")
        print(f"2. Headers: Content-Type=application/json, Authorization=Bearer {masked_key}, User-Agent=Mozilla/5.0...")
        print(f"3. Model Name: {self.model}")
        print(f"4. Request Body Payload (truncated): {json.dumps(payload)[:300]}...")

        req = urllib.request.Request(self.url, data=json.dumps(payload).encode('utf-8'), headers=headers)
        
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                res_text = resp.read().decode('utf-8')
                print(f"SUCCESS (HTTP {resp.status})")
                data = json.loads(res_text)
                return data['choices'][0]['message']['content']
        except urllib.error.HTTPError as e:
            err_body = e.read().decode('utf-8', errors='ignore')
            print(f"5. HTTP Response Code: {e.code} {e.reason}")
            print(f"6. Response Headers: {dict(e.headers)}")
            print(f"7. Response Body: {err_body}")
            print("=======================================================\n")
            raise e

class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = self._discover_model()
        print(f"Selected Gemini model:\n{self.model}")
        self.url = f"https://generativelanguage.googleapis.com/v1beta/{self.model}:generateContent?key={self.api_key}"

    def _discover_model(self) -> str:
        list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={self.api_key}"
        try:
            req = urllib.request.Request(list_url)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                models = data.get('models', [])
                for m in models:
                    supported = m.get('supportedGenerationMethods', [])
                    if "generateContent" in supported:
                        name = m.get('name', '')
                        if name:
                            return name
        except Exception as e:
            raise RuntimeError(f"Failed to list Gemini models using provided API key: {e}")

        raise RuntimeError(f"No Gemini model supporting 'generateContent' was found for the provided API key.")

    def generate(self, prompt: str, timeout: int = 15, max_retries: int = 2) -> str:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"}
        }
        masked_key = self.api_key[:6] + "..." + self.api_key[-4:] if len(self.api_key) > 10 else "***"
        masked_url = self.url.replace(self.api_key, masked_key)
        print(f"[GEMINI REQUEST] Final Endpoint URL: {masked_url}")

        req = urllib.request.Request(self.url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
        time.sleep(1.0)  # Pacing pause to prevent 429 Rate Limit
        res_text = self._execute_http_with_retry(req, timeout=timeout, max_retries=max_retries)
        data = json.loads(res_text)
        return data['candidates'][0]['content']['parts'][0]['text']

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = "https://api.openai.com/v1/chat/completions"

    def generate(self, prompt: str, timeout: int = 15, max_retries: int = 2) -> str:
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "system", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.1
        }
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {self.api_key}'})
        res_text = self._execute_http_with_retry(req, timeout=timeout, max_retries=max_retries)
        data = json.loads(res_text)
        return data['choices'][0]['message']['content']

class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = "https://api.anthropic.com/v1/messages"

    def generate(self, prompt: str, timeout: int = 15, max_retries: int = 2) -> str:
        payload = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json', 'x-api-key': self.api_key, 'anthropic-version': '2023-06-01'})
        res_text = self._execute_http_with_retry(req, timeout=timeout, max_retries=max_retries)
        data = json.loads(res_text)
        return data['content'][0]['text']

class LLMReasoner:
    """Hybrid AI Reasoning Module supporting provider abstraction, OpenRouter, and strict JSON parsing."""

    def __init__(self):
        self.provider: Optional[LLMProvider] = None
        self._initialize_provider()

    def _initialize_provider(self):
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        gemini_key = os.environ.get("GEMINI_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

        if openrouter_key:
            self.provider = OpenRouterProvider(openrouter_key)
            logger.info(f"LLM Reasoner initialized with OpenRouter Provider (Model: {self.provider.model}).")
        elif gemini_key:
            self.provider = GeminiProvider(gemini_key)
            logger.info("LLM Reasoner initialized with Gemini Provider.")
        elif openai_key:
            self.provider = OpenAIProvider(openai_key)
            logger.info("LLM Reasoner initialized with OpenAI Provider.")
        elif anthropic_key:
            self.provider = AnthropicProvider(anthropic_key)
            logger.info("LLM Reasoner initialized with Anthropic Provider.")
        else:
            logger.warning("No LLM API keys found in environment. LLM Reasoner will operate in baseline fallback mode.")

    def reason(
        self,
        context: MessageContext,
        features: Dict[str, Any],
        baseline_result: Optional[Dict[str, Any]],
        evidence_items: List[EvidenceDetail]
    ) -> Dict[str, Any]:
        
        msg = context.current_message
        curr_msg_payload = {
            "message_id": msg.message_id,
            "conversation_type": msg.conversation_type,
            "created_at": msg.created_at,
            "text": msg.message_text,
            "media_type": msg.media_type,
            "forwarded_count": msg.forwarded_count
        }

        evidence_payload = [
            {
                "message_id": e.message_id,
                "relevance_score": e.total_score,
                "signals": e.triggered_signals
            }
            for e in evidence_items
        ]

        personalization_payload = {
            "user_id": context.receiver.user_id if context.receiver else "",
            "dnd_window": context.receiver.do_not_disturb_window if context.receiver else "",
            "is_group_muted": features.get('is_group_muted', False),
            "is_opted_out": features.get('is_opted_out', False)
        }

        fallback_response = baseline_result if baseline_result else {
            "action": "digest" if features.get('is_dnd') else "notify",
            "message_type": "personal",
            "reason": "Baseline fallback routing based on context features.",
            "confidence": 0.70,
            "evidence_message_ids": [e.message_id for e in evidence_items]
        }

        if not self.provider:
            return fallback_response

        prompt = SYSTEM_PROMPT.format(
            current_message=json.dumps(curr_msg_payload),
            extracted_features=json.dumps(features),
            baseline_recommendation=json.dumps(baseline_result if baseline_result else {}),
            selected_evidence=json.dumps(evidence_payload),
            personalization_context=json.dumps(personalization_payload)
        )

        # File-based response caching by prompt hash
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".llm_cache")
        os.makedirs(cache_dir, exist_ok=True)
        prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
        cache_file = os.path.join(cache_dir, f"{prompt_hash}.json")

        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    return cached_data
            except Exception:
                pass

        try:
            raw_response = self.provider.generate(prompt, timeout=15, max_retries=4).strip()
            if raw_response.startswith("```json"):
                raw_response = raw_response[7:]
            if raw_response.startswith("```"):
                raw_response = raw_response[3:]
            if raw_response.endswith("```"):
                raw_response = raw_response[:-3]
            raw_response = raw_response.strip()

            parsed = json.loads(raw_response)
            
            required_keys = ['action', 'message_type', 'reason', 'confidence', 'evidence_message_ids']
            if not all(k in parsed for k in required_keys):
                logger.error(f"LLM JSON missing keys: {parsed}")
                return fallback_response

            ev_list = parsed.get('evidence_message_ids', [])
            if isinstance(ev_list, str):
                ev_list = [ev_list] if ev_list != "none" else []

            result = {
                "action": str(parsed['action']).lower(),
                "message_type": str(parsed['message_type']).lower(),
                "reason": str(parsed['reason']),
                "confidence": float(parsed['confidence']),
                "evidence_message_ids": ev_list
            }

            # Save to cache
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f)
            except Exception:
                pass

            return result

        except Exception as e:
            logger.error(f"LLM reasoning error: {e}. Falling back to baseline.")
            return fallback_response
