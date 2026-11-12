import json
import subprocess
import sys
import pytest

from main import reconcile, trust_and_impact, verify_claim, words


def test_no_evidence_is_unverifiable():
    result = verify_claim("The photo was taken at the summit", [])
    assert result["verdict"] == "unverifiable"
    assert result["cited_evidence"] == []


def test_one_match_is_uncertain():
    result = verify_claim("The photo was taken at the summit", ["Summit photo taken Monday", "Unrelated"])
    assert result["verdict"] == "uncertain"
    assert result["cited_evidence"] == ["ev_0"]


def test_two_matches_are_supported():
    result = verify_claim("The photo was taken at the summit", ["Summit photo", "Photo taken Monday"])
    assert result["verdict"] == "supported"
    assert result["confidence"] == pytest.approx(0.7)


def test_negation_in_evidence_results_in_contradicted():
    # Keywords "summit" and "photo" match, but "not" triggers contradiction
    result = verify_claim("summit photo", ["this is not a summit photo"])
    assert result["verdict"] == "contradicted"
    assert result["confidence"] < 0.5  # Downgraded from 0.5 (uncertain base) to 0.1


def test_negation_in_uncited_evidence_is_ignored():
    # An unrelated snippet containing negation should not contradict a matched snippet
    result = verify_claim("summit photo", ["summit photo verified", "unrelated debunked fake claim"])
    assert result["verdict"] == "uncertain"
    assert result["cited_evidence"] == ["ev_0"]


def test_stop_words_are_ignored():
    # "that" and "from" are stop words and should not cause a match
    result = verify_claim("that from", ["that from"])
    assert result["verdict"] == "unverifiable"


def test_stop_words_with_punctuation_are_ignored():
    # Punctuation attached to stop words should still be filtered out
    assert words("That, and this.") == set()
    assert words("... ! ?") == set()


def test_cited_ids_always_exist_in_evidence():
    result = verify_claim("summit photograph", ["summit view", "photograph archive", "nothing"])
    assert set(result["cited_evidence"]) <= set(result["evidence"])


@pytest.mark.parametrize("claim", ["", "   ", "x" * 501])
def test_bad_claim_is_rejected(claim):
    with pytest.raises(ValueError):
        verify_claim(claim, [])


def test_trust_and_impact_formulas():
    result = trust_and_impact(aigc=0.2, belief=0.8, dissemination=0.3)
    assert result["trust"] == pytest.approx(0.6)
    assert result["impact"] == pytest.approx(1.1)
    assert result["trust_level"] == "high_trust"
    assert result["impact_level"] == "moderate_impact"


def test_believed_ai_content_is_suspicious_and_viral():
    result = trust_and_impact(aigc=0.9, belief=0.9, dissemination=0.9)
    assert result["trust_level"] == "suspicious"
    assert result["impact_level"] == "viral_risk"


@pytest.mark.parametrize("scores", [(-0.1, 0.5, 0.5), (0.5, 1.1, 0.5)])
def test_out_of_range_scores_are_rejected(scores):
    with pytest.raises(ValueError):
        trust_and_impact(*scores)


def test_reconciliation_dangerous_disconnect():
    claim_check = {"verdict": "supported", "confidence": 0.7}
    human_response = {"trust_level": "low_trust", "impact_level": "low_impact"}
    assert reconcile(claim_check, human_response) == "dangerous_disconnect"


def test_reconciliation_viral_misinformation():
    claim_check = {"verdict": "contradicted", "confidence": 0.1}
    human_response = {"trust_level": "suspicious", "impact_level": "viral_risk"}
    assert reconcile(claim_check, human_response) == "viral_misinformation"


def test_reconciliation_debunked():
    claim_check = {"verdict": "contradicted", "confidence": 0.1}
    human_response = {"trust_level": "low_trust", "impact_level": "low_impact"}
    assert reconcile(claim_check, human_response) == "debunked"


def test_reconciliation_grounded_truth():
    claim_check = {"verdict": "supported", "confidence": 0.7}
    human_response = {"trust_level": "high_trust", "impact_level": "high_impact"}
    assert reconcile(claim_check, human_response) == "grounded_truth"


def test_reconciliation_plausible_synthetic_viral():
    claim_check = {"verdict": "uncertain", "confidence": 0.5}
    human_response = {"trust_level": "suspicious", "impact_level": "viral_risk"}
    assert reconcile(claim_check, human_response) == "plausible_synthetic_viral"


def test_reconciliation_benign_hallucination():
    claim_check = {"verdict": "unverifiable", "confidence": 0.0}
    human_response = {"trust_level": "low_trust", "impact_level": "low_impact"}
    assert reconcile(claim_check, human_response) == "benign_hallucination"


def test_cli_end_to_end():
    cmd = [
        sys.executable,
        "main.py",
        "--claim", "The photo was taken at the summit",
        "--evidence", "Summit photo taken Monday",
        "--evidence", "Photo taken at dawn",
        "--scores", "0.2", "0.8", "0.6",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    output = json.loads(proc.stdout)

    assert output["claim_check"]["verdict"] == "supported"
    assert "human_response" in output
    assert output["reconciliation"] == "grounded_truth"
