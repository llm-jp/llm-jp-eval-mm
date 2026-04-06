"""Shared utilities for MMMU and JMMMU scorers.

This module contains the common logic extracted from mmmu_scorer.py and
jmmmu_scorer.py.  Language-specific differences (prompts, domain categories,
parse_open_response indicators, extract_numbers patterns) remain in the
respective scorer files.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict

import numpy as np
from datasets import Dataset

from .scorer import AggregateOutput

# Maximum number of per-question images supported by the MMMU dataset.
MAX_IMAGES_PER_QUESTION = 7


# ---------------------------------------------------------------------------
# Token / option helpers
# ---------------------------------------------------------------------------

def replace_images_tokens(input_string: str) -> str:
    for i in range(1, MAX_IMAGES_PER_QUESTION + 1):
        question_text = f"<image {i}>"
        query_text = "<image>"
        if question_text in input_string:
            input_string = input_string.replace(question_text, query_text)
    return input_string


def parse_options(options: list[str]) -> str:
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join(
        [
            f"{option_letter}. {option}"
            for option_letter, option in zip(option_letters, options)
        ]
    )
    return choices_str


def construct_prompt(doc: dict, multi_choice_prompt: str, open_ended_prompt: str) -> str:
    question = doc["question"]
    question = question.replace("<image1>", "<image 1>")
    if doc["question_type"] == "multiple-choice":
        # Weirdly, data["options"] is a string in MMMU Huggingface dataset
        parsed_options = parse_options(ast.literal_eval(doc["options"]))
        question = f"{question}\n{parsed_options}\n\n{multi_choice_prompt}"
    else:
        question = f"{question}\n\n{open_ended_prompt}"
    return question


# ---------------------------------------------------------------------------
# Subset name extraction
# ---------------------------------------------------------------------------

def extract_subset_name(input_string: str) -> str:
    split = input_string.split("_")[0]
    pattern = re.compile(rf"^{split}_(.+?)_\d+$")
    match = pattern.search(input_string)
    if match:
        return match.group(1)
    else:
        raise ValueError(f'No match found in "{input_string}"')


# ---------------------------------------------------------------------------
# Multi-choice response parsing
# ---------------------------------------------------------------------------

def parse_multi_choice_response(
    response: str,
    all_choices: list[str],
    index2ans: dict[str, str],
    random_choice: bool,
) -> str | None:
    """Parse the prediction from the generated response.

    Return the predicted index e.g., A, B, C, D.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10
    """
    for char in [",", ".", "!", "?", ";", ":", "'", "\u3001", "\u3002", "\uff01", "\uff1f", "\uff1b", "\uff1a"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match
    japanese_char_pattern = r"[\u3040-\u30FF\u4E00-\u9FFF]"
    index_ans = True
    ans_with_brack = False
    candidates: list[str] = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f"{choice} " in response:
                candidates.append(choice)

    # Between the Japanese characters
    if len(candidates) == 0:
        for choice in all_choices:
            pattern = rf"{japanese_char_pattern}{choice}{japanese_char_pattern}"
            if re.search(pattern, response):
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A. B. C. D.
            if f"{choice}." in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:
        if random_choice:
            pred_index = np.random.choice(all_choices)
        else:
            pred_index = None
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    idx = response.rfind(f"({can})")
                    start_indexes.append(idx)
            else:
                for can in candidates:
                    idx = response.rfind(f" {can} ")
                    start_indexes.append(idx)
        else:
            for can in candidates:
                idx = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(idx)
        pred_index = candidates[np.argmax(start_indexes)]
    else:
        pred_index = candidates[0]

    return pred_index


# ---------------------------------------------------------------------------
# Number / string normalisation
# ---------------------------------------------------------------------------

def check_is_number(string: str) -> bool:
    """Check if the given string a number.

    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L65
    """
    try:
        float(string.replace(",", ""))
        return True
    except ValueError:
        return False


def normalize_str(string: str) -> list:
    """Normalize the str to lower case and make them float numbers if possible.

    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L76
    """
    string = string.strip()
    is_number = check_is_number(string)

    if is_number:
        string = string.replace(",", "")
        string = float(string)
        string = round(string, 2)
        return [string]
    else:
        string = string.lower()
        if len(string) == 1:
            return [" " + string, string + " "]  # avoid trivial matches
        return [string]


def get_multi_choice_info(options: list[str]) -> tuple[dict[str, str], list[str]]:
    """Given the list of options for multiple choice question
    Return the index2ans and all_choices.

    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/data_utils.py#L54
    """
    start_chr = "A"
    all_choices: list[str] = []
    index2ans: dict[str, str] = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))
    return index2ans, all_choices


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def calculate_ins_level_acc(results: dict) -> float:
    """Calculate the instruction level accuracy for given Subject results.

    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L246
    """
    acc = 0
    ins_num = 0
    for cat_results in results.values():
        acc += cat_results["acc"] * cat_results["num_example"]
        ins_num += cat_results["num_example"]
    if ins_num == 0:
        return 0
    return acc / ins_num


def eval_multi_choice(gold_i, pred_i) -> bool:
    """Evaluate a multiple choice instance.

    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L175
    """
    correct = False
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    else:
        if gold_i == pred_i:
            correct = True
    return correct


def eval_open(gold_i, pred_i) -> bool:
    """Evaluate an open question instance.

    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L191
    """
    correct = False
    if isinstance(gold_i, list):
        norm_answers = []
        for answer in gold_i:
            norm_answers.extend(normalize_str(answer))
    else:
        norm_answers = normalize_str(gold_i)
    for pred in pred_i:
        if isinstance(pred, str):
            for norm_ans in norm_answers:
                if isinstance(norm_ans, str) and norm_ans in pred:
                    if not correct:
                        correct = True
                    break
        else:
            if pred in norm_answers:
                if not correct:
                    correct = True
                break
    return correct


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def get_score(
    doc: Dataset,
    pred: str,
    random_choice: bool,
    extract_numbers_fn,
    open_response_indicators: list[str] | None = None,
) -> int:
    """Score a single prediction against a document.

    Parameters
    ----------
    doc : Dataset row
    pred : model prediction string
    random_choice : whether to randomly pick when no candidate found
    extract_numbers_fn : language-specific number extraction function
    open_response_indicators : key sub-response indicators (defaults to English)
    """
    if doc["question_type"] == "multiple-choice":
        index2ans, all_choices = get_multi_choice_info(ast.literal_eval(doc["options"]))
        parsed_pred = parse_multi_choice_response(
            pred, all_choices, index2ans, random_choice
        )
    else:
        parsed_pred = parse_open_response(pred, extract_numbers_fn, open_response_indicators)
    answer = doc["answer"]
    question_type = doc["question_type"]
    if question_type == "multiple-choice":
        correct = eval_multi_choice(answer, parsed_pred)
    else:
        correct = eval_open(answer, parsed_pred)

    return int(correct)


def parse_open_response(response: str, extract_numbers_fn, indicators: list[str] | None = None) -> list:
    """Parse the prediction from the generated response.

    Return a list of predicted strings or numbers.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L122

    Parameters
    ----------
    response : model response text
    extract_numbers_fn : language-specific number extraction callable
    indicators : key sub-response indicator strings (defaults to English indicators)
    """
    if indicators is None:
        indicators = [
            "could be ",
            "so ",
            "is ",
            "thus ",
            "therefore ",
            "final ",
            "answer ",
            "result ",
        ]

    def get_key_subresponses(response):
        key_responses = []
        response = response.strip().strip("\u3002")
        sub_responses = re.split(r"[\u3002\uff01\uff1f.]\s*|\n", response)

        key_responses = []
        for index, resp in enumerate(sub_responses):
            if index == len(sub_responses) - 1:
                indicators.extend(["\uff1d", "="])
            shortest_key_response = None
            for indicator in indicators:
                if indicator in resp:
                    if not shortest_key_response:
                        shortest_key_response = resp.split(indicator)[-1].strip()
                    else:
                        if len(resp.split(indicator)[-1].strip()) < len(
                            shortest_key_response
                        ):
                            shortest_key_response = resp.split(indicator)[-1].strip()

            if shortest_key_response:
                if shortest_key_response.strip() not in [
                    ",", ".", "!", "?", ";", ":", "'",
                    "\u3001", "\u3002", "\uff01", "\uff1f", "\uff1b", "\uff1a",
                ]:
                    key_responses.append(shortest_key_response)
        if len(key_responses) == 0:
            return [response]
        return key_responses

    key_responses = get_key_subresponses(response)

    pred_list = key_responses.copy()
    for resp in key_responses:
        pred_list.extend(extract_numbers_fn(resp))

    tmp_pred_list = []
    for i in range(len(pred_list)):
        tmp_pred_list.extend(normalize_str(pred_list[i]))
    pred_list = tmp_pred_list

    # remove duplicates
    pred_list = list(set(pred_list))

    return pred_list


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_mmmu_results(
    docs,
    scores: list[int],
    domain_cat2sub_cat: dict[str, list[str]],
) -> AggregateOutput:
    """Shared aggregation logic for MMMU and JMMMU."""
    evaluation_result = {}
    subset_to_eval_samples = defaultdict(list)
    for doc, score in zip(docs, scores):
        subdomain = extract_subset_name(doc["id"])
        subset_to_eval_samples[subdomain].append(score)
    for subset, sub_eval_samples in subset_to_eval_samples.items():
        num = len(sub_eval_samples)
        acc = sum(sub_eval_samples) / num
        evaluation_result[subset] = {"num_example": num, "acc": acc}
    printable_results: dict[str, float] = {}
    for domain, in_domain_cats in domain_cat2sub_cat.items():
        in_domain_cat_results = {}
        for cat_name in in_domain_cats:
            if cat_name in evaluation_result.keys():
                in_domain_cat_results[cat_name] = evaluation_result[cat_name]
        in_domain_ins_acc = calculate_ins_level_acc(in_domain_cat_results)
        printable_results["Overall-" + domain] = round(in_domain_ins_acc, 5)
        for cat_name, cat_results in in_domain_cat_results.items():
            printable_results[cat_name] = round(cat_results["acc"], 5)
    all_ins_acc = calculate_ins_level_acc(evaluation_result)
    printable_results["Overall"] = round(all_ins_acc, 5)
    return AggregateOutput(all_ins_acc, printable_results)
