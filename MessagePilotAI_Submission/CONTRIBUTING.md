# Contributing to MessagePilot AI

Thank you for your interest in contributing to MessagePilot AI!

## Development Workflow
1. Fork the repository and create a feature branch (`git checkout -b feature/amazing-feature`).
2. Run the 10-point automated QA test suite before committing:
   ```bash
   python code/qa_test_suite.py
   ```
3. Ensure no API keys or absolute system paths are committed.
4. Push your branch and open a Pull Request.

## Code Standards
- All Python code must conform to PEP 8 style guidelines.
- Additive scorecard modifications in `code/rule_engine.py` must maintain zero false notifications.
- Any new features must include relative path resolution for 100% portability.
