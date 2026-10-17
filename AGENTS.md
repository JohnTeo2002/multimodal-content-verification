# Repository Guidance for AI Agents

This repo is being rebuilt from a single file into a small offline PyTorch and agent system. The plan is in [issues.md](issues.md), which lists 15 issues, their dependencies and their owners. Follow it. Read the issue before you change its files.

## Layout (target)

```
config/default_config.yaml   central settings (issue 14)
core/                        activations, backbones, cross-attention, dual-branch verifier (issues 2-5)
cognitive/                   HR-MCP fusion, propensity and composite metrics (issues 6, 15)
agents/                      Pydantic schemas, AgentFact, async MCP mock server (issues 7-9)
tests/                       one test file per area, pytest only (issue 10)
main.py                      argparse CLI: one-shot run and --serve (issue 13)
```

Until issue 12 lands, `README.md` still describes the current single-file tool. Do not treat it as the spec.

## Rules for every code change

- Python 3.12, type hints everywhere, one job per class or function.
- Keep the tensor shape contracts written in each issue, and comment shapes at each step.
- Keep the system offline. No network calls, no checkpoint or weight downloads, and no real API keys in files. External services are stubs marked `TODO-EXTENSION-MARKER [NAME]`, injectable so tests never touch the network.
- Keep settings in `config/default_config.yaml`. Every key must be read by code, and no key may be unused.
- Validate input at trust boundaries: CLI arguments, MCP tool inputs and config loading. Limit image size so one request cannot exhaust memory.
- Citations in `agents/` come only from the evidence pool. A verdict must never be `supported` or `true` on stub evidence.
- MCP tools are async. Errors come back as structured `isError` results, not exceptions.
- Each change adds or updates tests in `tests/`. Run `pytest -q` before you finish.
- Keep to the smallest change that meets the issue. Don't add features the issue doesn't ask for, and don't wire the cross-attention layer into the models unless an issue says so.

## Owners (from issues.md)

- John: skeleton, cross-modal attention, HR-MCP fusion, schemas, MCP server, tests, CI, docs, CLI, config.
- Isaac: activations, backbones, dual-branch verifier.
- Liu Bin: AgentFact orchestrator, propensity and credibility metrics.

If an issue's owner is unclear, ask before changing its files.
