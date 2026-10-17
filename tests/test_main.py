import pytest

from main import trust_and_impact, verify_claim


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
