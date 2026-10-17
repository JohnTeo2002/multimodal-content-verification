# Repository Guidance for AI Agents

Use this file as the default operating guide for work in this repository. Keep changes small, offline-friendly, and aligned with the existing module boundaries.

## Start Here

- Read [README.md](README.md) for the project overview and the supported demo commands.
- Read [architecture_decisions.md](architecture_decisions.md) before changing cross-cutting behavior or public interfaces.
- Use [pytest.ini](pytest.ini) as the test discovery source of truth.

## Core Commands

- Install dependencies with `pip install -r requirements.txt`.
- Run the demo pipeline with `python main.py`.
- Run the async MCP-style server mode with `python main.py --serve`.
- Run tests with `pytest`.

## Project Boundaries

- `core/` owns the PyTorch verifier, mocked backbones, activation functions, and fusion logic.
- `cognitive/` owns HR-MCP fusion and propensity/composite metrics.
- `agents/` owns evidence-grounded agent orchestration, Pydantic schemas, and async tool dispatch.
- `main.py` is the CLI and top-level pipeline wiring layer.

## Editing Rules

- Preserve tensor shape contracts and Pydantic validation paths when changing model code.
- Keep MCP handlers async and return structured errors across the tool boundary.
- Keep the system offline by default; do not add live network calls, pretrained checkpoints, or transport integrations unless explicitly requested.
- Keep configuration centralized in [config/default_config.yaml](config/default_config.yaml).
- Add a new entry to [architecture_decisions.md](architecture_decisions.md) for changes that affect architecture, interfaces, or operational behavior.

## Practical Pitfalls

- The repository is a demo/reference implementation, so synthetic inputs and random weights are expected unless a task explicitly replaces them.
- `main.py` currently wires only the supported demo paths; do not assume unused YAML settings are automatically consumed.
- Evidence grounding matters in `agents/`; keep citation IDs and validation rules intact.

## When In Doubt

- Prefer the smallest local change that preserves the existing design.
- Link to existing docs instead of repeating them here.