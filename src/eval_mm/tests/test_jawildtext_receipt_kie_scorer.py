import json

import pytest

from eval_mm.metrics.jawildtext_receipt_kie_scorer import (
    JaWildTextReceiptKIEScorer,
    _extract_json_from_response,
    _flatten_to_tokens,
    _normalize_kie_value,
)
from eval_mm.metrics.scorer import ScorerConfig


@pytest.fixture
def scorer():
    return JaWildTextReceiptKIEScorer(ScorerConfig())


def _make_gold(**kwargs) -> str:
    return json.dumps(kwargs, ensure_ascii=False)


class TestNormalizeKieValue:
    def test_remove_yen(self):
        assert _normalize_kie_value("¥1,000") == "1000"

    def test_remove_fullwidth_yen(self):
        assert _normalize_kie_value("￥500") == "500"

    def test_nfkc(self):
        assert _normalize_kie_value("１２３") == "123"

    def test_lowercase(self):
        assert _normalize_kie_value("ABC") == "abc"

    def test_none_input(self):
        assert _normalize_kie_value(None) == ""


class TestExtractJson:
    def test_raw_json(self):
        text = '{"store_name": "テスト"}'
        result = _extract_json_from_response(text)
        assert result == {"store_name": "テスト"}

    def test_markdown_code_block(self):
        text = '```json\n{"store_name": "テスト"}\n```'
        result = _extract_json_from_response(text)
        assert result == {"store_name": "テスト"}

    def test_json_with_surrounding_text(self):
        text = 'Here is the result: {"store_name": "テスト"} end'
        result = _extract_json_from_response(text)
        assert result == {"store_name": "テスト"}

    def test_invalid_json(self):
        result = _extract_json_from_response("no json here")
        assert result is None


class TestFlattenToTokens:
    def test_scalar_fields(self):
        data = {"store_name": "テスト店", "date": "2026-01-01"}
        tokens = _flatten_to_tokens(data)
        assert tokens["store_name:テスト店"] == 1
        assert tokens["date:2026-01-01"] == 1

    def test_line_items(self):
        data = {
            "line_items": [
                {"item_name": "商品A", "item_price": "100", "item_quantity": "1"}
            ]
        }
        tokens = _flatten_to_tokens(data)
        assert tokens["line_items.item_name:商品a"] == 1
        assert tokens["line_items.item_price:100"] == 1
        assert tokens["line_items.item_quantity:1"] == 1

    def test_none_values_excluded(self):
        data = {"store_name": None, "date": "2026-01-01"}
        tokens = _flatten_to_tokens(data)
        assert "store_name:" not in tokens
        assert tokens["date:2026-01-01"] == 1


class TestScore:
    def test_exact_match(self, scorer):
        gold = _make_gold(store_name="テスト店", date="2026-01-01")
        pred = '{"store_name": "テスト店", "date": "2026-01-01"}'
        scores = scorer.score([gold], [pred])
        assert scores[0]["f1"] == 1.0
        assert scores[0]["parse_error"] is False

    def test_partial_match(self, scorer):
        gold = _make_gold(store_name="テスト店", date="2026-01-01")
        pred = '{"store_name": "テスト店"}'
        scores = scorer.score([gold], [pred])
        assert 0.0 < scores[0]["f1"] < 1.0
        assert scores[0]["precision"] == 1.0
        assert scores[0]["recall"] < 1.0

    def test_parse_error(self, scorer):
        gold = _make_gold(store_name="テスト店")
        pred = "this is not json"
        scores = scorer.score([gold], [pred])
        assert scores[0]["f1"] == 0.0
        assert scores[0]["parse_error"] is True

    def test_yen_normalization(self, scorer):
        gold = _make_gold(total_amount="¥1,000")
        pred = '{"total_amount": "1000"}'
        scores = scorer.score([gold], [pred])
        assert scores[0]["f1"] == 1.0

    def test_field_accuracy(self, scorer):
        gold = _make_gold(store_name="テスト店", date="2026-01-01")
        pred = '{"store_name": "テスト店", "date": "2026-01-02"}'
        scores = scorer.score([gold], [pred])
        assert scores[0]["field_accuracy"]["field_store_name"] == 1.0
        assert scores[0]["field_accuracy"]["field_date"] == 0.0

    def test_markdown_json(self, scorer):
        gold = _make_gold(store_name="テスト店")
        pred = '```json\n{"store_name": "テスト店"}\n```'
        scores = scorer.score([gold], [pred])
        assert scores[0]["f1"] == 1.0

    def test_multiple_samples(self, scorer):
        gold1 = _make_gold(store_name="A")
        gold2 = _make_gold(store_name="B")
        pred1 = '{"store_name": "A"}'
        pred2 = "not json"
        scores = scorer.score([gold1, gold2], [pred1, pred2])
        assert len(scores) == 2
        assert scores[0]["f1"] == 1.0
        assert scores[1]["parse_error"] is True


class TestAggregate:
    def test_aggregate(self, scorer):
        scores = [
            {
                "f1": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "field_accuracy": {f"field_{f}": 1.0 for f in [
                    "store_name", "store_address", "receipt_id",
                    "date", "time", "total_amount", "tax_amount",
                ]},
                "parse_error": False,
            },
            {
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "field_accuracy": {f"field_{f}": 0.0 for f in [
                    "store_name", "store_address", "receipt_id",
                    "date", "time", "total_amount", "tax_amount",
                ]},
                "parse_error": True,
            },
        ]
        result = scorer.aggregate(scores)
        assert result.overall_score == pytest.approx(0.5)
        assert result.details["f1"] == pytest.approx(0.5)
        assert result.details["parse_error_count"] == 1
        # Field accuracy only from non-parse-error samples
        assert result.details["field_store_name"] == 1.0

    def test_aggregate_empty(self, scorer):
        result = scorer.aggregate([])
        assert result.overall_score == 0.0
