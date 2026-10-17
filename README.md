# Content Verification Tool

A lightweight command-line tool written in Python that provides two basic analytical checks for text content:
1. **Claim Verification:** Compares a claim against supplied evidence using keyword matching.
2. **Human Response Scoring:** Computes trust and impact metrics based on AI-generation, belief, and dissemination scores.

---

## 1. What This Project Does

This project provides a simple CLI utility to evaluate claims against text evidence and calculate trust and impact scores.

### Problems It Solves
- Quickly checks whether text evidence snippets share common keywords with a given claim to find supporting context.
- Calculates basic trust and potential impact levels from numerical scores representing how AI-generated content appears, how believable it is, and how likely it is to spread.

### Important Limitations
- **Not Real AI/ML:** The claim checker uses crude keyword matching (words longer than 3 characters) rather than semantic understanding or factual verification.
- **No Contradiction Detection:** Sharing words results in a positive match (`supported`), even if the evidence contradicts the claim.
- **Rule-Based Formulas:** Trust and impact levels are calculated using arithmetic differences/sums against fixed thresholds rather than real-world machine learning models.

---

## 2. How It Works

The application runs via a command-line interface (`main.py`) using Python's standard library (`argparse` and `json`). 

1. **Input Parsing:** Receives a `--claim`, zero or more `--evidence` snippets, and optional `--scores`.
2. **Claim Check (`verify_claim`):**
   - Extracts words longer than 3 characters from both the claim and each evidence snippet (ignoring case and punctuation).
   - Identifies which evidence snippets share matching words with the claim.
   - Calculates a confidence score based on the number of matching evidence sources (capped at `0.9`).
   - Assigns a verdict (`unverifiable`, `uncertain`, or `supported`) based on whether matches were found and the confidence level.
3. **Human Response Scoring (`trust_and_impact`):**
   - Validates that all three input scores are between `0` and `1`.
   - Calculates `trust` (`belief - aigc`) and `impact` (`belief + dissemination`).
   - Determines qualitative levels (e.g., `high_trust`, `suspicious`, `viral_risk`, `moderate_impact`).
4. **Output:** Prints a formatted JSON object containing the results.

---

## 3. Mermaid Diagram

```mermaid
graph TD
    A[CLI Input: --claim, --evidence, --scores] --> B{Parse & Validate Arguments}
    B --> C[Claim Check: verify_claim]
    C --> D[Extract Words > 3 chars & Match]
    D --> E[Calculate Confidence & Verdict]
    B --> F[Human Response Scoring: trust_and_impact]
    F --> G[Calculate Trust & Impact Formulas]
    G --> H[Determine Qualitative Levels]
    E --> I[Output JSON Result]
    H --> I
```

---

## 4. Repository Structure

```text
multimodal-content-verification/
├── main.py            # Main command-line script containing verification and scoring logic
├── requirements.txt   # Project dependencies (pytest for testing)
├── pytest.ini         # Configuration settings for pytest
└── tests/
    └── test_main.py   # Unit tests verifying claim matching and scoring formulas
```

---

## 5. Setup

### Prerequisites
- Python 3.8 or higher installed on your system.

### Installation Steps

1. Clone or open the repository directory:
   ```bash
   cd multimodal-content-verification
   ```

2. Create a Python virtual environment:
   ```bash
   python3 -m venv venv
   ```

3. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows (Command Prompt):
     ```cmd
     venv\Scripts\activate.bat
     ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 6. How to Run the Project

### Simplest Working Command (Claim Check Only)
```bash
python3 main.py --claim "The photo was taken at the summit" --evidence "Summit photo taken Monday" --evidence "Photo taken at dawn"
```

### CLI Arguments
- `--claim`: The statement or claim text you want to check (required, 1 to 500 characters).
- `--evidence`: Supporting evidence text snippet. You can pass this argument multiple times to supply multiple pieces of evidence (optional).
- `--scores`: Three floating-point numbers between `0` and `1` representing `AIGC` (how AI-generated it looks), `BELIEF` (how believed it is), and `DISSEMINATION` (how likely to be shared) respectively (optional).

### Comprehensive Example with Scores
```bash
python3 main.py --claim "The photo was taken at the summit" \
    --evidence "Summit photo taken Monday" \
    --evidence "Photo taken at dawn" \
    --scores 0.2 0.8 0.8
```

### Example Output Meaning
```json
{
  "claim_check": {
    "verdict": "supported",
    "confidence": 0.7,
    "cited_evidence": [
      "ev_0",
      "ev_1"
    ],
    "evidence": {
      "ev_0": "Summit photo taken Monday",
      "ev_1": "Photo taken at dawn"
    }
  },
  "human_response": {
    "trust": 0.6,
    "trust_level": "high_trust",
    "impact": 1.4,
    "impact_level": "viral_risk"
  }
}
```
- **`verdict`**: Set to `supported` because multiple evidence snippets shared matching words with the claim.
- **`confidence`**: Calculated confidence level (e.g., `0.7`).
- **`cited_evidence`**: IDs of evidence snippets that matched (`ev_0`, `ev_1`).
- **`trust` / `impact`**: Numerical calculations derived from the input scores.
- **`trust_level` / `impact_level`**: Qualitative labels categorized by threshold rules.

---

## 7. How to Run the Tests

Run the test suite using `pytest`:
```bash
pytest
```

### What the Tests Verify
- **Claim Verification (`tests/test_main.py`):** Checks that zero evidence yields `unverifiable`, single matches yield `uncertain`, multiple matches yield `supported`, valid evidence IDs are cited, and blank or overly long claims are rejected with a `ValueError`.
- **Scoring Formulas:** Verifies correct calculations for trust and impact, boundary limits (rejection of scores outside `0` to `1`), and threshold classifications (e.g., suspicious/viral risk flags).

---

## 8. Current Limitations

- **Keyword Matching Limitation:** Uses basic word intersection (`len(w) > 3`), meaning words with different contexts or negations may trigger a positive match.
- **No Factual Ground Truth:** The tool does not connect to external databases, search engines, or real-world knowledge bases.
- **Synthetic Scoring:** Human response metrics are purely arithmetic formulas and threshold rules, not actual telemetry or human survey data.
