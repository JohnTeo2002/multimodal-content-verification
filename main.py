"""Check a claim against supplied evidence, and score human-response metrics.

    python main.py --claim "The photo was taken at the summit" \
        --evidence "Summit photo taken Monday" --scores 0.2 0.8 0.6

This is keyword matching, not fact-checking: "supported" only means the
evidence shares words with the claim. It cannot detect contradiction.
"""

import argparse
import json

MAX_CLAIM_LENGTH = 500


def words(text):
    # shortcut: words of 3 letters or fewer are ignored, replace with real NLP if matching is too crude
    return {w.lower().strip(".,!?\"'") for w in text.split() if len(w) > 3}


def verify_claim(claim, evidence):
    """evidence is a list of snippets. Returns the verdict and the evidence ids it cites."""
    if not claim.strip() or len(claim) > MAX_CLAIM_LENGTH:
        raise ValueError(f"claim must be 1-{MAX_CLAIM_LENGTH} characters and not blank")

    pool = {f"ev_{i}": snippet for i, snippet in enumerate(evidence)}
    cited = [id for id, snippet in pool.items() if words(claim) & words(snippet)]

    # More matching evidence gives more confidence, capped so it never claims certainty.
    confidence = min(0.9, 0.3 + 0.2 * len(cited)) if cited else 0.0
    if not cited:
        verdict = "unverifiable"
    elif confidence < 0.6:
        verdict = "uncertain"
    else:
        verdict = "supported"
    return {"verdict": verdict, "confidence": confidence, "cited_evidence": cited, "evidence": pool}


def trust_and_impact(aigc, belief, dissemination):
    """Each score is 0 to 1: how AI-generated it looks, how believed, how likely to be shared."""
    for score in (aigc, belief, dissemination):
        if not 0 <= score <= 1:
            raise ValueError("scores must be between 0 and 1")

    trust = round(belief - aigc, 6)
    impact = round(belief + dissemination, 6)

    if aigc >= 0.7 and belief >= 0.7:
        trust_level = "suspicious"
    elif trust >= 0.5:
        trust_level = "high_trust"
    elif trust >= 0:
        trust_level = "moderate_trust"
    else:
        trust_level = "low_trust"

    if belief >= 0.7 and dissemination >= 0.7:
        impact_level = "viral_risk"
    elif impact >= 1.3:
        impact_level = "high_impact"
    elif impact >= 0.7:
        impact_level = "moderate_impact"
    else:
        impact_level = "low_impact"

    return {"trust": trust, "trust_level": trust_level, "impact": impact, "impact_level": impact_level}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--claim", required=True)
    parser.add_argument("--evidence", action="append", default=[], help="evidence text, repeat for more")
    parser.add_argument("--scores", type=float, nargs=3, metavar=("AIGC", "BELIEF", "DISSEMINATION"))
    args = parser.parse_args()

    try:
        result = {"claim_check": verify_claim(args.claim, args.evidence)}
        if args.scores:
            result["human_response"] = trust_and_impact(*args.scores)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
