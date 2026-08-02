# HackerRank Orchestrate — WhatsApp AI Notification Router

A production-grade, multimodal, hybrid AI notification routing engine built for WhatsApp. It processes text messages, OCR poster/screenshot images, and ASR voice notes to intelligently classify each message into **`notify`**, **`digest`**, or **`mute`**.

---

## 🚀 Quick Start & How to Run

### 1. Execute Main Pipeline (Generates `dataset/output.csv`)
```bash
python code/main.py
```

### 2. Run Evaluation Benchmark (Against `sample_messages.csv`)
```bash
python code/evaluate.py
```

---

## 🏗️ Architecture & Pipeline Overview

```text
Incoming Message (Text / Image OCR / Voice ASR)
                      │
                      ▼
            Context Builder & Loader
                      │
                      ▼
              Feature Extractor
                      │
                      ▼
         Stage 1 & 2 Evidence Retriever
                      │
                      ▼
         Deterministic Rule Engine (Confidence >= 0.90)
                      │
           ┌──────────┴──────────┐
           │                     │
   High Confidence        Low Confidence
           │                     │
           ▼                     ▼
     Bypass LLM            LLM Reasoner (MD5 Caching)
           │                     │
           └──────────┬──────────┘
                      │
                      ▼
          Confidence Calibrator [0.35, 0.98]
                      │
                      ▼
     Decision Trace Logger & dataset/output.csv
```

---

## 📊 Solution Highlights

1. **Multimodal Processing**:
   - **OCR Subsystem**: Primary `EasyOCR` with `Pytesseract` fallback to extract structured text, QR code/payment flags, and poster categories.
   - **ASR Subsystem**: Primary `Faster-Whisper` (`int8`, `beam_size=5`) with OpenAI `Whisper` fallback to transcribe voice notes.

2. **Two-Stage Evidence Retriever**:
   - **Stage 1**: Candidate filtering by user, group, business, and sender.
   - **Stage 2**: Additive scoring across 10 signals (same sender/group/biz, exact/near-duplicate Jaccard similarity, recency exponential decay, past user reports/replies/mutes/opens). Returns top 5 evidence.

3. **Hybrid AI Router**:
   - High-confidence rules ($\ge 0.90$) bypass LLM execution to minimize API cost and eliminate latency.
   - Low-confidence or ambiguous messages route to provider-abstracted LLM reasoner (Gemini, OpenRouter, OpenAI, Anthropic).

4. **Response Caching**:
   - MD5 prompt hashing stores LLM responses in `.llm_cache/` to ensure zero duplicate API calls and instant reproducible evaluation.

5. **Production Confidence Calibration**:
   - Weighted formula: $0.40 \times \text{Rule} + 0.30 \times \text{Evidence} + 0.20 \times \text{Personalization} + 0.10 \times \text{LLM}$.
   - Clamped strictly between `[0.35, 0.98]`.

6. **Decision Trace Logging**:
   - Every prediction generates a detailed JSON trace in `decision_traces.jsonl` (separate from `output.csv`) for explainability during AI Judge interviews.

---

## 📄 Output Schema (`dataset/output.csv`)

| Column | Description |
|---|---|
| `message_id` | Unique ID of incoming message |
| `action` | Routing decision (`notify`, `digest`, `mute`) |
| `message_type` | Category (`personal`, `urgent`, `event`, `payment`, `business_update`, `promotion`, `greeting`, `forward`, `spam`, `scam`, `unknown`) |
| `reason` | One-sentence human-readable explanation |
| `confidence` | Calibrated float score (`0.35` – `0.98`) |
| `evidence_message_ids` | Semicolon-separated evidence IDs or `none` |

---

## 🔧 Environment Setup & API Keys

Environment variables supported (optional for LLM reasoning):
- `GEMINI_API_KEY`
- `OPENROUTER_API_KEY`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
