import pytest

from eval_mm.metrics.jawildtext_handwriting_ocr_scorer import (
    JaWildTextHandwritingOCRScorer,
    _levenshtein,
    _normalize_ocr_text,
)
from eval_mm.metrics.scorer import ScorerConfig


@pytest.fixture
def scorer():
    return JaWildTextHandwritingOCRScorer(ScorerConfig())


class TestNormalizeOcrText:
    def test_nfkc_normalization(self):
        # Full-width digits → half-width
        assert _normalize_ocr_text("１２３") == "123"

    def test_lowercase(self):
        assert _normalize_ocr_text("ABC") == "abc"

    def test_escaped_newline_to_real(self):
        assert _normalize_ocr_text("hello\\nworld") == "hello\nworld"

    def test_collapse_spaces(self):
        assert _normalize_ocr_text("a   b") == "a b"

    def test_empty_string(self):
        assert _normalize_ocr_text("") == ""

    def test_none_input(self):
        assert _normalize_ocr_text(None) == ""

    def test_crlf_to_lf(self):
        assert _normalize_ocr_text("a\r\nb") == "a\nb"


class TestLevenshtein:
    def test_identical(self):
        assert _levenshtein("abc", "abc") == 0

    def test_empty_both(self):
        assert _levenshtein("", "") == 0

    def test_empty_one(self):
        assert _levenshtein("abc", "") == 3
        assert _levenshtein("", "abc") == 3

    def test_substitution(self):
        assert _levenshtein("abc", "axc") == 1

    def test_insertion(self):
        assert _levenshtein("ac", "abc") == 1

    def test_deletion(self):
        assert _levenshtein("abc", "ac") == 1


class TestScore:
    def test_exact_match(self, scorer):
        scores = scorer.score(["こんにちは"], ["こんにちは"])
        assert scores == [1.0]

    def test_completely_wrong(self, scorer):
        scores = scorer.score(["abc"], ["xyz"])
        assert scores[0] == 0.0

    def test_partial_match(self, scorer):
        scores = scorer.score(["こんにちは世界"], ["こんにちわ世界"])
        assert 0.0 < scores[0] < 1.0

    def test_nfkc_equivalence(self, scorer):
        # Full-width vs half-width should normalize to same
        scores = scorer.score(["123"], ["１２３"])
        assert scores == [1.0]

    def test_newline_handling(self, scorer):
        scores = scorer.score(["hello\nworld"], ["hello\\nworld"])
        assert scores == [1.0]

    def test_empty_ref_empty_pred(self, scorer):
        scores = scorer.score([""], [""])
        assert scores == [1.0]

    def test_empty_ref_nonempty_pred(self, scorer):
        scores = scorer.score([""], ["some text"])
        assert scores == [0.0]

    def test_nonempty_ref_empty_pred(self, scorer):
        scores = scorer.score(["some text"], [""])
        assert scores == [0.0]

    def test_case_insensitive(self, scorer):
        scores = scorer.score(["Hello World"], ["hello world"])
        assert scores == [1.0]

    def test_multiple_samples(self, scorer):
        refs = ["abc", "xyz"]
        preds = ["abc", "abc"]
        scores = scorer.score(refs, preds)
        assert len(scores) == 2
        assert scores[0] == 1.0
        assert scores[1] == 0.0


class TestAggregate:
    def test_aggregate_mean(self, scorer):
        result = scorer.aggregate([1.0, 0.5, 0.0])
        assert result.overall_score == pytest.approx(0.5)
        assert result.details["cer_similarity"] == pytest.approx(0.5)

    def test_aggregate_empty(self, scorer):
        result = scorer.aggregate([])
        assert result.overall_score == 0.0
