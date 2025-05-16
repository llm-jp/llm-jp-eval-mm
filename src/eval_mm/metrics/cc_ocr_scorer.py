import re
from collections import Counter
from typing import List, Dict, Any, cast  # Added cast for type hinting clarity

from .scorer import Scorer, AggregateOutput, ScorerConfig


def token_normalize(
    token_text: str, is_lower: bool = False, is_alphanum_only: bool = False
) -> str:
    """
    Normalizes a single token.
    - Converts to lowercase if is_lower is True.
    - Removes non-alphanumeric characters if is_alphanum_only is True.
    """
    if is_lower:
        token_text = token_text.lower()
    if is_alphanum_only:
        token_text = re.sub("[^A-Za-z0-9]+", "", token_text)
    return token_text


def text_normalize_and_tokenize(
    text: str,
    is_keep_blank: bool = True,
    is_lower: bool = True,
    is_alphanum_only: bool = False,
) -> List[str]:
    """
    Normalizes and tokenizes a text string.
    - Replaces tabs, newlines, and specific markers (###, ***).
    - Reduces multiple spaces to a single space.
    - If is_keep_blank is False, removes all spaces.
    - Splits into tokens: by space if is_keep_blank is True, otherwise character by character.
    - Normalizes each token using token_normalize.
    - Filters out empty tokens.
    """
    text = str(
        text
    ).strip()  # Ensure text is a string and strip leading/trailing whitespace
    text = (
        text.replace("\t", " ").replace("\n", " ").replace("###", "").replace("***", "")
    )
    text = re.sub(r"\s+", " ", text)  # Reduce multiple spaces to one
    if not is_keep_blank:
        text = text.replace(
            " ", ""
        )  # Remove all spaces if not keeping blanks (char level)

    text_tokens = text.split(" ") if is_keep_blank else list(text)

    text_token_normalized = [
        token_normalize(t, is_lower, is_alphanum_only) for t in text_tokens
    ]
    text_token_normalized = [x for x in text_token_normalized if len(x) > 0]
    return text_token_normalized


def evaluate_single_sample(gts: List[str], preds: List[str]) -> int:
    """
    Calculates the number of correctly matched tokens between ground truth and prediction lists.
    This is based on token counts (similar to bag-of-words comparison).
    """
    right_num = 0
    gt_counter_info = dict(Counter(gts))
    pdt_counter_info = dict(Counter(preds))
    for gt_token, gt_count in gt_counter_info.items():
        pred_count = pdt_counter_info.get(gt_token, 0)
        right_num += min(gt_count, pred_count)
    return right_num


def calculate_metrics(
    response_info: Dict[str, List[str]],
    gt_info: Dict[str, List[str]],
    is_verbose: bool = False,
) -> Dict[str, float]:
    """
    Calculates macro and micro averaged Precision, Recall, and F1-score.
    - response_info: Dictionary ожидающий format {'id': list_of_pred_tokens, ...}
    - gt_info: Dictionary ожидающий format {'id': list_of_gt_tokens, ...}
    - is_verbose: If True, returns all metrics; otherwise, returns only macro_f1 and micro_f1.
    """
    macro_recall_list: List[float] = []
    macro_precision_list: List[float] = []
    macro_f1_list: List[float] = []
    total_gt_num, total_pred_num, total_right_num = 0, 0, 0

    if not gt_info:  # Handle empty ground truth
        if is_verbose:
            return {
                "macro_recall": 0.0,
                "macro_precision": 0.0,
                "macro_f1_score": 0.0,
                "micro_recall": 0.0,
                "micro_precision": 0.0,
                "micro_f1_score": 0.0,
            }
        else:
            return {"macro_f1_score": 0.0, "micro_f1_score": 0.0}

    for file_name, fullbox_gts in gt_info.items():
        fullbox_preds = response_info.get(file_name, [])
        right_num = evaluate_single_sample(fullbox_gts, fullbox_preds)
        total_right_num += right_num
        current_gt_len = len(fullbox_gts)
        current_pred_len = len(fullbox_preds)
        total_gt_num += current_gt_len
        total_pred_num += current_pred_len

        macro_recall = right_num / (current_gt_len + 1e-9)
        macro_precision = right_num / (current_pred_len + 1e-9)
        macro_f1 = (
            2 * macro_recall * macro_precision / (macro_recall + macro_precision + 1e-9)
        )
        macro_recall_list.append(macro_recall)
        macro_precision_list.append(macro_precision)
        macro_f1_list.append(macro_f1)

    # Macro average calculation
    final_macro_recall = (
        sum(macro_recall_list) / (len(macro_recall_list) + 1e-9)
        if macro_recall_list
        else 0.0
    )
    final_macro_precision = (
        sum(macro_precision_list) / (len(macro_precision_list) + 1e-9)
        if macro_precision_list
        else 0.0
    )
    final_macro_f1 = (
        sum(macro_f1_list) / (len(macro_f1_list) + 1e-9) if macro_f1_list else 0.0
    )

    # Micro average calculation
    recall_acc = total_right_num / (total_gt_num + 1e-9) if total_gt_num > 0 else 0.0
    preci_acc = total_right_num / (total_pred_num + 1e-9) if total_pred_num > 0 else 0.0
    hmean = (
        2 * recall_acc * preci_acc / (recall_acc + preci_acc + 1e-9)
        if (recall_acc + preci_acc) > 0
        else 0.0
    )

    vbs_eval_result = {
        "macro_recall": final_macro_recall,
        "macro_precision": final_macro_precision,
        "macro_f1_score": final_macro_f1,
        "micro_recall": recall_acc,
        "micro_precision": preci_acc,
        "micro_f1_score": hmean,
    }
    eval_result = (
        vbs_eval_result
        if is_verbose
        else {"macro_f1_score": final_macro_f1, "micro_f1_score": hmean}
    )
    return eval_result


# CCOCRScorer class, specialized for Japanese (character-level, no alphanum_only)
class CCOCRScorer(Scorer):
    def __init__(self, config: ScorerConfig):
        super().__init__(config)
        # Settings specialized for Japanese text evaluation:
        # - Character-level tokenization (is_word_level = False)
        # - No restriction to alphanumeric characters (is_alphanum_only = False)
        # - Convert to lowercase (is_lower = True), mainly affects Latin characters if present.
        self.is_word_level: bool = False
        self.is_alphanum_only: bool = False
        self.is_lower: bool = True  # Retained True as in original OCR evaluator logic

        # This will store tokenized data from the `score` method for use in `aggregate`.
        self._processed_data_for_aggregation: List[Dict[str, Any]] = []

    def score(self, refs: List[str], preds: List[str]) -> List[float]:
        self._processed_data_for_aggregation = []  # Clear previous data
        sample_f1_scores: List[float] = []

        for i, (ref_text, pred_text) in enumerate(zip(refs, preds)):
            # text_normalize_and_tokenize uses is_word_level to determine is_keep_blank.
            # For character-level (Japanese), is_keep_blank should be False.
            gt_tokens = text_normalize_and_tokenize(
                ref_text,
                is_keep_blank=self.is_word_level,  # False for char level
                is_lower=self.is_lower,
                is_alphanum_only=self.is_alphanum_only,
            )
            pred_tokens = text_normalize_and_tokenize(
                pred_text,
                is_keep_blank=self.is_word_level,  # False for char level
                is_lower=self.is_lower,
                is_alphanum_only=self.is_alphanum_only,
            )

            # Store tokenized data for the aggregate method
            self._processed_data_for_aggregation.append(
                {
                    "id": f"sample_{i}",  # ID for matching in calculate_metrics
                    "gt_tokens": gt_tokens,
                    "pred_tokens": pred_tokens,
                }
            )

            # Calculate F1 score for the current sample
            right_num = evaluate_single_sample(gt_tokens, pred_tokens)

            gt_len = len(gt_tokens)
            pred_len = len(pred_tokens)

            recall = right_num / (gt_len + 1e-9) if gt_len > 0 else 0.0
            precision = right_num / (pred_len + 1e-9) if pred_len > 0 else 0.0
            f1 = (
                2 * recall * precision / (recall + precision + 1e-9)
                if (recall + precision) > 0
                else 0.0
            )

            sample_f1_scores.append(f1)

        return sample_f1_scores

    def aggregate(self, scores: List[float]) -> AggregateOutput:
        # overall_score is the mean of per-sample F1 scores
        overall_score = sum(scores) / len(scores) if scores else 0.0

        details: Dict[str, Any] = {"mean_sample_f1": overall_score}

        # Calculate detailed metrics using data stored from the score method
        if self._processed_data_for_aggregation:
            # Prepare data in the format expected by calculate_metrics
            response_info = {
                s["id"]: s["pred_tokens"] for s in self._processed_data_for_aggregation
            }
            gt_info = {
                s["id"]: s["gt_tokens"] for s in self._processed_data_for_aggregation
            }

            # Get all metrics by setting is_verbose=True
            calculated_metrics = calculate_metrics(
                response_info, gt_info, is_verbose=True
            )
            details.update(calculated_metrics)
        else:  # Ensure metrics are present even if no data was processed
            empty_metrics = calculate_metrics({}, {}, is_verbose=True)
            details.update(empty_metrics)

        # Include the fixed metric configuration in the details
        details["metric_config"] = {
            "is_word_level": self.is_word_level,
            "is_lower": self.is_lower,
            "is_alphanum_only": self.is_alphanum_only,
            "description": "Optimized for Japanese (character-level)",
        }

        return AggregateOutput(overall_score, cast(Dict[str, float], details))


def test_cc_ocr_scorer():
    # ScorerConfig might be needed by the base Scorer class.
    # Provide a dummy one if its content is not relevant for CCOCRScorer's direct logic.
    config = ScorerConfig()

    scorer = CCOCRScorer(config)

    # Test 1: Exact match (Japanese example)
    refs1 = ["これはテストです。", "第二の例です。"]
    preds1 = ["これはテストです。", "第二の例です。"]
    # Expected: F1 = 1.0 for each sample, as they are character-level exact matches.
    scores1 = scorer.score(refs1, preds1)
    assert all(
        abs(s - 1.0) < 1e-6 for s in scores1
    ), f"Test 1 Scores: Expected all 1.0, got {scores1}"
    agg_output1 = scorer.aggregate(scores1)
    assert abs(agg_output1.overall_score - 1.0) < 1e-6, "Test 1 Overall Score"
    assert abs(agg_output1.details["micro_f1_score"] - 1.0) < 1e-6, "Test 1 Micro F1"
    assert abs(agg_output1.details["macro_f1_score"] - 1.0) < 1e-6, "Test 1 Macro F1"
    print("Test 1 (Exact Match - Japanese) Passed!")

    # Test 2: Partial match (Japanese example, character-level)
    # Sample 1:
    # ref: "リンゴ バナナ オレンジ"
    #   -> normalized by text_normalize_and_tokenize(..., is_keep_blank=False, ...)
    #   -> "リンゴバナナオレンジ"
    #   -> gt_tokens1 = ['リ', 'ン', 'ゴ', 'バ', 'ナ', 'ナ', 'オ', 'レ', 'ン', 'ジ'] (10 characters)
    # pred: "リンゴ バ ナナ"
    #   -> normalized by text_normalize_and_tokenize(..., is_keep_blank=False, ...)
    #   -> "リンゴバナナ"
    #   -> pred_tokens1 = ['リ', 'ン', 'ゴ', 'バ', 'ナ', 'ナ'] (6 characters)
    # Matching tokens (evaluate_single_sample):
    #   Counter(gt_tokens1): {'リ':1, 'ン':2, 'ゴ':1, 'バ':1, 'ナ':2, 'オ':1, 'レ':1, 'ジ':1}
    #   Counter(pred_tokens1): {'リ':1, 'ン':1, 'ゴ':1, 'バ':1, 'ナ':2}
    #   right_num1: min(gt('リ'),pred('リ'))=1 + min(gt('ン'),pred('ン'))=1 (pred 'ン' is 1) + ... = 6
    #   (Specifically: 'リ':1, 'ン':1, 'ゴ':1, 'バ':1, 'ナ':2 -> Total: 1+1+1+1+2 = 6)
    # R1 (Recall) = right_num1 / len(gt_tokens1) = 6 / 10 = 0.6
    # P1 (Precision) = right_num1 / len(pred_tokens1) = 6 / 6 = 1.0
    # F1_1 = 2 * (R1 * P1) / (R1 + P1) = 2 * (0.6 * 1.0) / (0.6 + 1.0) = 1.2 / 1.6 = 0.75

    # Sample 2:
    # ref: "猫 犬"
    #   -> normalized "猫犬"
    #   -> gt_tokens2 = ['猫', '犬'] (2 characters)
    # pred: "猫 犬 鳥"
    #   -> normalized "猫犬鳥"
    #   -> pred_tokens2 = ['猫', '犬', '鳥'] (3 characters)
    # Matching tokens (evaluate_single_sample):
    #   Counter(gt_tokens2): {'猫':1, '犬':1}
    #   Counter(pred_tokens2): {'猫':1, '犬':1, '鳥':1}
    #   right_num2: min(gt('猫'),pred('猫'))=1 + min(gt('犬'),pred('犬'))=1 = 2
    # R2 (Recall) = right_num2 / len(gt_tokens2) = 2 / 2 = 1.0
    # P2 (Precision) = right_num2 / len(pred_tokens2) = 2 / 3
    # F1_2 = 2 * (R2 * P2) / (R2 + P2) = 2 * (1.0 * 2/3) / (1.0 + 2/3) = (4/3) / (5/3) = 4/5 = 0.8

    refs2 = ["リンゴ バナナ オレンジ", "猫 犬"]
    preds2 = ["リンゴ バ ナナ", "猫 犬 鳥"]
    scores2 = scorer.score(refs2, preds2)
    expected_scores2 = [0.75, 0.8]
    for s, es in zip(scores2, expected_scores2):
        assert abs(s - es) < 1e-6, f"Test 2 Sample Score: Expected {es}, got {s}"

    agg_output2 = scorer.aggregate(scores2)
    # Overall score (mean of sample F1s): (0.75 + 0.8) / 2 = 0.775
    assert (
        abs(agg_output2.overall_score - 0.775) < 1e-6
    ), f"Test 2 Overall Score: Expected 0.775, Got {agg_output2.overall_score}"

    # Micro F1:
    # total_gt_num = len_gt1 (10) + len_gt2 (2) = 12
    # total_pred_num = len_pred1 (6) + len_pred2 (3) = 9
    # total_right_num = right1 (6) + right2 (2) = 8
    # micro_R = total_right_num / total_gt_num = 8 / 12 = 2/3
    # micro_P = total_right_num / total_pred_num = 8 / 9
    # micro_F1 = 2 * (micro_R * micro_P) / (micro_R + micro_P)
    #          = 2 * ( (2/3) * (8/9) ) / ( (2/3) + (8/9) )
    #          = 2 * (16/27) / ( (6/9) + (8/9) )
    #          = 2 * (16/27) / (14/9)
    #          = (32/27) * (9/14) = 32 / (3 * 14) = 32 / 42 = 16/21
    #          (16/21 is approx 0.76190476)
    expected_micro_f1_2 = 16 / 21
    assert (
        abs(agg_output2.details["micro_f1_score"] - expected_micro_f1_2) < 1e-6
    ), f"Test 2 Micro F1: Expected {expected_micro_f1_2}, Got {agg_output2.details['micro_f1_score']}"

    # Macro F1: (F1_sample1 + F1_sample2) / 2 = (0.75 + 0.8) / 2 = 0.775
    expected_macro_f1_2 = 0.775
    assert (
        abs(agg_output2.details["macro_f1_score"] - expected_macro_f1_2) < 1e-6
    ), f"Test 2 Macro F1: Expected {expected_macro_f1_2}, Got {agg_output2.details['macro_f1_score']}"

    print("Test 2 (Partial Match - Japanese, Char Level) Passed!")

    # Test 3: Mismatch and normalization (Japanese)
    # text_normalize_and_tokenize handles spaces and some full-width/half-width considerations implicitly by char list.
    # Spaces are removed because is_keep_blank is False (due to is_word_level=False).
    refs3 = [
        "こんにちは　世界",
        "吾輩は猫である。名前はまだ無い。",
    ]  # Full-width space in ref1
    preds3 = [
        "こんにちわ世界",
        "吾輩は猫である。名前はまだ無 い。",
    ]  # Mistake in pred1, half-width space in pred2

    # Sample 1:
    # ref1: "こんにちは　世界"
    #   -> normalized: "こんにちは世界"
    #   -> gt_tokens1 = ['こ', 'ん', 'に', 'ち', 'は', '世', '界'] (7 characters)
    # pred1: "こんにちわ世界"
    #   -> normalized: "こんにちわ世界"
    #   -> pred_tokens1 = ['こ', 'ん', 'に', 'ち', 'わ', '世', '界'] (7 characters)
    # Matching: 'こ','ん','に','ち','世','界' (6 chars). 'は' vs 'わ' is a mismatch.
    # right_num1 = 6
    # R1 = 6/7, P1 = 6/7
    # F1_s1 = 2 * (6/7 * 6/7) / (6/7 + 6/7) = (72/49) / (12/7) = 72/49 * 7/12 = (6*1)/(7*1) = 6/7
    f1_s1 = 6 / 7  # approx 0.857142857

    # Sample 2:
    # ref2: "吾輩は猫である。名前はまだ無い。"
    #   -> normalized: "吾輩は猫である。名前はまだ無い。"
    #   -> gt_tokens2 = ['吾', '輩', 'は', '猫', 'で', 'あ', 'る', '。', '名', '前', 'は', 'ま', 'だ', '無', 'い', '。'] (16 characters)
    # pred2: "吾輩は猫である。名前はまだ無 い。"
    #   -> normalized: "吾輩は猫である。名前はまだ無い。" (space is removed)
    #   -> pred_tokens2 = ['吾', '輩', 'は', '猫', 'で', 'あ', 'る', '。', '名', '前', 'は', 'ま', 'だ', '無', 'い', '。'] (16 characters)
    # These become identical after normalization.
    # right_num2 = 16
    # R2 = 16/16 = 1.0, P2 = 16/16 = 1.0
    # F1_s2 = 1.0
    f1_s2 = 1.0

    scores3 = scorer.score(refs3, preds3)
    expected_scores3 = [f1_s1, f1_s2]
    for s, es in zip(scores3, expected_scores3):
        assert abs(s - es) < 1e-6, f"Test 3 Sample Score: Expected {es}, got {s}"

    agg_output3 = scorer.aggregate(scores3)
    # Overall score (mean of sample F1s) = ( (6/7) + 1.0 ) / 2 = ( (6/7) + (7/7) ) / 2 = (13/7) / 2 = 13/14
    expected_overall3 = 13 / 14  # approx 0.928571428
    assert (
        abs(agg_output3.overall_score - expected_overall3) < 1e-6
    ), f"Test 3 Overall Score: Expected {expected_overall3}, Got {agg_output3.overall_score}"

    # Micro F1 for Test 3:
    # s1: gt_len1=7, pred_len1=7, right1=6
    # s2: gt_len2=16, pred_len2=16, right2=16
    # total_gt = gt_len1 + gt_len2 = 7 + 16 = 23
    # total_pred = pred_len1 + pred_len2 = 7 + 16 = 23
    # total_right = right1 + right2 = 6 + 16 = 22
    # micro_R = total_right / total_gt = 22 / 23
    # micro_P = total_right / total_pred = 22 / 23
    # micro_F1 = 2 * (micro_R * micro_P) / (micro_R + micro_P)
    #          = 2 * ( (22/23) * (22/23) ) / ( (22/23) + (22/23) )
    #          = (2 * (22/23)^2) / (2 * (22/23)) = 22/23
    #          (22/23 is approx 0.956521739)
    expected_micro_f1_3 = 22 / 23
    assert (
        abs(agg_output3.details["micro_f1_score"] - expected_micro_f1_3) < 1e-6
    ), f"Test 3 Micro F1: Expected {expected_micro_f1_3}, Got {agg_output3.details['micro_f1_score']}"

    # Macro F1 for Test 3: (F1_sample1 + F1_sample2) / 2 = ( (6/7) + 1.0 ) / 2 = 13/14
    expected_macro_f1_3 = 13 / 14  # Same as overall_score for two samples
    assert (
        abs(agg_output3.details["macro_f1_score"] - expected_macro_f1_3) < 1e-6
    ), f"Test 3 Macro F1: Expected {expected_macro_f1_3}, Got {agg_output3.details['macro_f1_score']}"
    print("Test 3 (Mismatch and Normalization - Japanese) Passed!")

    # Test 4: Empty input
    refs_empty: List[str] = []
    preds_empty: List[str] = []
    scores_empty = scorer.score(refs_empty, preds_empty)
    assert scores_empty == [], "Test 4 Scores Empty"
    agg_output_empty = scorer.aggregate(scores_empty)
    assert agg_output_empty.overall_score == 0.0, "Test 4 Overall Score Empty"
    assert agg_output_empty.details["micro_f1_score"] == 0.0, "Test 4 Micro F1 Empty"
    assert agg_output_empty.details["macro_f1_score"] == 0.0, "Test 4 Macro F1 Empty"
    print("Test 4 (Empty input) Passed!")


if __name__ == "__main__":
    print("CCOCRScorer tests (Japanese specialized) starting...")
    test_cc_ocr_scorer()
    print("All CCOCRScorer tests finished!")
