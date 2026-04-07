import re
from collections import Counter
from typing import List, Dict, Any, cast  # Added cast for type hinting clarity

from .scorer import Scorer, AggregateOutput, ScorerConfig
from .scorer_registry import register_scorer


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
@register_scorer("cc-ocr")
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
