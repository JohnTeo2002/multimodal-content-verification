# Repository Guidance for AI Agents

Small, offline, single-file project. Keep it that way.

- Code is in `main.py`, tests in `tests/test_main.py`. Run `python main.py --help` and `pytest`.
- Keep input validation (claim length, score range) and the rule that cited evidence IDs come only from the supplied evidence.
- No new dependencies, network calls or extra layers unless asked.
- See [README.md](README.md) for usage.
