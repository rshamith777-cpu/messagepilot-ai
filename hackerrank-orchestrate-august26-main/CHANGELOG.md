# Changelog

All notable changes to the MessagePilot AI project will be documented in this file.

## [v3.0.0] - 2026-08-02
### Added
- Okapi BM25 ranking algorithm with length-ratio pre-filtered fuzzy sequence matching.
- `UserMemoryEngine` tracking user interaction rates (open, reply, mute, report).
- Large-scale synthetic benchmark generator (`benchmark/synthetic_generator.py`) for 1,000 synthetic test messages.
- Systems analytics engine (`benchmark/analytics_engine.py`) calculating throughput and $p_{95}$ latency distributions.
- Multi-tab interactive explainability dashboard (`dashboard/index.html`).
- Comprehensive Technical Systems Specification Report (`docs/TECHNICAL_REPORT.md`).

## [v2.0.0] - 2026-08-02
### Added
- Additive Hybrid Scorecard Engine (`rule_engine.py`) computing explicit $S_{\text{notify}}, S_{\text{digest}}, S_{\text{mute}}$ vectors.
- Mathematical score-gap confidence calibration formula clamped to $[0.35, 0.98]$.
- Internal 20-scenario unseen edge-case benchmark suite (`benchmark/run_benchmark.py`).
- Structured Decision Graph logging to `decision_traces.jsonl`.

## [v1.0.0] - 2026-08-01
### Added
- Initial release with multimodal EasyOCR / Faster-Whisper pre-processing.
- Baseline rule engine and LLM reasoner with MD5 prompt caching.
- 10-point automated QA test suite (`code/qa_test_suite.py`).
