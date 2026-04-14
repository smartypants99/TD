from timedilate.scorer import Scorer


def test_build_scoring_prompt():
    scorer = Scorer()
    prompt = scorer.build_scoring_prompt("Write a sort function", "def sort(lst): return sorted(lst)")
    assert "sort" in prompt.lower()
    assert "elegance" in prompt.lower()


def test_parse_score_valid():
    scorer = Scorer()
    assert scorer.parse_score("85") == 85
    assert scorer.parse_score("Score: 72") == 72
    assert scorer.parse_score("I'd give this a 90 out of 100") == 90


def test_parse_score_invalid():
    scorer = Scorer()
    assert scorer.parse_score("no number here") == 0
    assert scorer.parse_score("") == 0


def test_parse_score_clamps():
    scorer = Scorer()
    assert scorer.parse_score("150") == 100
    assert scorer.parse_score("-5") == 5  # regex extracts digits, no negative scores exist
