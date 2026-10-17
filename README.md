# Content Verification

A small command-line tool with two checks:

1. **Claim check:** compares a claim to evidence text you supply by matching words. The verdict is `supported`, `uncertain` or `unverifiable`, and it lists which evidence it cited. It cannot detect contradiction, so `supported` only means the evidence shares words with the claim.
2. **Human response:** turns three scores between 0 and 1 (how AI-generated it looks, how believed, how likely to be shared) into trust and impact, with a label for each.

```bash
python main.py --claim "The photo was taken at the summit" \
    --evidence "Summit photo taken Monday" --evidence "Photo taken at dawn" \
    --scores 0.2 0.8 0.6
pytest
```

Everything is in `main.py`. The tests are in `tests/test_main.py`. There are no dependencies except `pytest`.

The earlier version, with fake image models, a mock MCP server and a YAML config, is in git history at commit `bdb0f9b`.
