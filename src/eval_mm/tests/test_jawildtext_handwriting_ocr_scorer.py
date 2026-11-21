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

    def test_case_preserved(self):
        assert _normalize_ocr_text("ABC") == "ABC"

    def test_escaped_newline_to_real(self):
        assert _normalize_ocr_text("hello\\nworld") == "hello\nworld"

    def test_remove_spaces(self):
        assert _normalize_ocr_text("a   b") == "ab"

    def test_remove_tabs_and_full_width_spaces(self):
        assert _normalize_ocr_text("電荷\tq\u3000=\u00a01.6") == "電荷q=1.6"

    def test_line_breaks_kept_spaces_around_removed(self):
        assert _normalize_ocr_text("  一行目 \n 二行目  ") == "一行目\n二行目"

    def test_empty_string(self):
        assert _normalize_ocr_text("") == ""

    def test_none_input(self):
        assert _normalize_ocr_text(None) == ""

    def test_crlf_to_lf(self):
        assert _normalize_ocr_text("a\r\nb") == "a\nb"

    def test_escaped_newline_before_japanese(self):
        assert _normalize_ocr_text("です\\nまた") == "です\nまた"

    def test_escaped_newline_before_word_starting_with_command(self):
        # "\\nu" is a LaTeX command, "\\nunit" is a line break before "unit"
        assert _normalize_ocr_text("a\\nunit") == "a\nunit"
        assert _normalize_ocr_text("a\\neko") == "a\neko"

    @pytest.mark.parametrize(
        "latex",
        [r"\nabla p", r"a \neq b", r"\nu", r"\not\in", r"a \ne b", r"\nabla^2 v"],
    )
    def test_latex_commands_kept(self, latex):
        assert "\n" not in _normalize_ocr_text(latex)

    def test_latex_and_line_break_mixed(self):
        text = r"-\nabla p + \nu\nabla^2 v\n次の行"
        assert _normalize_ocr_text(text) == "-\\nablap+\\nu\\nabla^2v\n次の行"


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

    def test_case_sensitive(self, scorer):
        scores = scorer.score(["Hello World"], ["hello world"])
        assert scores[0] == pytest.approx(1 - 2 / len("HelloWorld"))

    def test_spaces_ignored(self, scorer):
        ref = "電荷q=1.6×10⁻¹⁹の電子が\n磁場0.025Tを垂直に横切る"
        pred = "電荷 q = 1.6×10⁻¹⁹ の電子が\n磁場 0.025 T を垂直に横切る"
        assert scorer.score([ref], [pred]) == [1.0]

    def test_line_breaks_count(self, scorer):
        scores = scorer.score(["一行目\n二行目"], ["一行目二行目"])
        assert scores[0] == pytest.approx(1 - 1 / len("一行目\n二行目"))

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
