# Reference: https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/tasks/jmmmu/utils.py

from datasets import Dataset, load_dataset, concatenate_datasets

from ..api.registry import register_task
from ..api.task import Task


import ast
import random
import re
from collections import defaultdict

import numpy as np


MULTI_CHOICE_PROMPT = (
    "与えられた選択肢の中から最も適切な回答のアルファベットを直接記入してください。"
)
OPEN_ENDED_PROMPT = "質問に対する回答を単語や短いフレーズで記入してください。"

DOMAIN_CAT2SUB_CAT = {
    "Art and Psychology": ["Design", "Music", "Psychology"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science": [
        "Biology",
        "Chemistry",
        "Math",
        "Physics",
    ],
    "Health and Medicine": [
        "Basic_Medical_Science",
        "Clinical_Medicine",
        "Diagnostics_and_Laboratory_Medicine",
        "Pharmacy",
        "Public_Health",
    ],
    "Tech and Engineering": [
        "Agriculture",
        "Architecture_and_Engineering",
        "Computer_Science",
        "Electronics",
        "Energy_and_Power",
        "Materials",
        "Mechanical_Engineering",
    ],
}


def replace_images_tokens(input_string):
    for i in range(1, 8):
        question_text = f"<image {i}>"
        query_text = "<image>"
        if question_text in input_string:
            input_string = input_string.replace(question_text, query_text)
    return input_string


def parse_options(options):
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join(
        [
            f"{option_letter}. {option}"
            for option_letter, option in zip(option_letters, options)
        ]
    )
    return choices_str


def construct_prompt(doc):
    question = doc["question"]
    question = question.replace("<image1>", "<image 1>")
    if doc["question_type"] == "multiple-choice":
        # Weirdly, data["options"] is a string in MMMU Huggingface dataset
        parsed_options = parse_options(ast.literal_eval(doc["options"]))
        # parsed_options already prepends a newline so no need to add space here
        question = f"{question}\n{parsed_options}\n\n{MULTI_CHOICE_PROMPT}"
    else:
        question = f"{question}\n\n{OPEN_ENDED_PROMPT}"
    return question


def jmmmu_doc_to_text(doc):
    question = construct_prompt(doc)
    question = replace_images_tokens(question)  # TODO: check if this is necessary
    return question


def jmmmu_doc_to_visual(doc):
    prompt = construct_prompt(doc)

    image_tokens = re.findall(r"<image \d+>", prompt)
    # Remove <> and  swap space as _
    image_tokens = sorted(
        list(
            set(
                [
                    image_token.strip("<>").replace(" ", "_")
                    for image_token in image_tokens
                ]
            )
        )
    )
    visual = [doc[image_token].convert("RGB") for image_token in image_tokens]
    return visual


def jmmmu_process_results(doc, results):
    pred = results[0]
    if doc["question_type"] == "multiple-choice":
        index2ans, all_choices = get_multi_choice_info(ast.literal_eval(doc["options"]))
        parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)
    else:
        parsed_pred = parse_open_response(pred)
    id = doc["id"]
    jmmmu_acc = {
        "id": id,
        "subdomain": extract_subset_name(doc["id"]),
        "question_type": doc["question_type"],
        "answer": doc["answer"],
        "parsed_pred": parsed_pred,
    }
    return {
        "jmmmu_acc": jmmmu_acc,
        "submission": {
            id: pred,
        },
    }


def extract_subset_name(input_string):
    # Define a regex pattern to match "validation_" at the beginning and "_<number>" at the end
    split = input_string.split("_")[0]
    pattern = re.compile(rf"^{split}_(.+?)_\d+$")
    match = pattern.search(input_string)
    if match:
        return match.group(1)
    else:
        raise ValueError(f'No match found in "{input_string}"')


def jmmmu_aggregate_results(results):
    evaluation_result = {}
    subset_to_eval_samples = defaultdict(list)
    for result in results:
        subset_to_eval_samples[result["subdomain"]].append(result)
    for subset, sub_eval_samples in subset_to_eval_samples.items():
        judge_dict, metric_dict = evaluate_jmmmu(sub_eval_samples)
        metric_dict.update({"num_example": len(sub_eval_samples)})
        evaluation_result[subset] = metric_dict
    printable_results = {}
    for domain, in_domain_cats in DOMAIN_CAT2SUB_CAT.items():
        in_domain_cat_results = {}
        for cat_name in in_domain_cats:
            if cat_name in evaluation_result.keys():
                in_domain_cat_results[cat_name] = evaluation_result[cat_name]
            else:
                pass
        in_domain_ins_acc = calculate_ins_level_acc(in_domain_cat_results)
        in_domain_data_num = sum(
            [
                cat_results["num_example"]
                for cat_results in in_domain_cat_results.values()
            ]
        )
        printable_results["Overall-" + domain] = {
            "num": int(in_domain_data_num),
            "acc": round(in_domain_ins_acc, 5),
        }
        # add sub category
        for cat_name, cat_results in in_domain_cat_results.items():
            printable_results[cat_name] = {
                "num": int(cat_results["num_example"]),
                "acc": round(cat_results["acc"], 5),
            }
    all_ins_acc = calculate_ins_level_acc(evaluation_result)
    printable_results["Overall"] = {
        "num": sum(
            [cat_results["num_example"] for cat_results in evaluation_result.values()]
        ),
        "acc": round(all_ins_acc, 5),
    }
    return printable_results


##################
# Helper functions written by official MMMU repo.
##################


def calculate_ins_level_acc(results):
    """Calculate the instruction level accuracy for given Subject results
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


def eval_multi_choice(gold_i, pred_i):
    """
    Evaluate a multiple choice instance.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L175
    """
    correct = False
    # only they are exactly the same, we consider it as correct
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    else:  # gold_i is a string
        if gold_i == pred_i:
            correct = True
    return correct


def eval_open(gold_i, pred_i):
    """
    Evaluate an open question instance
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L191
    """
    correct = False
    if isinstance(gold_i, list):
        # use float to avoid trivial matches
        norm_answers = []
        for answer in gold_i:
            norm_answers.extend(normalize_str(answer))
    else:
        norm_answers = normalize_str(gold_i)
    for pred in pred_i:  # pred is already normalized in parse response phase
        if isinstance(pred, str):  # if it's a string, then find if ans in the pred_i
            for norm_ans in norm_answers:
                # only see if the string answer in the string pred
                if isinstance(norm_ans, str) and norm_ans in pred:
                    if not correct:
                        correct = True
                    break
        else:  # it's a float number
            if pred in norm_answers:
                if not correct:
                    correct = True
                break
    return correct


def evaluate_jmmmu(samples):
    """
    Batch evaluation for multiple choice and open questions.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L219
    """
    pred_correct = 0
    judge_dict = dict()
    for sample in samples:
        gold_i = sample["answer"]
        pred_i = sample["parsed_pred"]
        if sample["question_type"] == "multiple-choice":
            correct = eval_multi_choice(gold_i, pred_i)
        else:  # open question
            correct = eval_open(gold_i, pred_i)

        if correct:
            judge_dict[sample["id"]] = "Correct"
            pred_correct += 1
        else:
            judge_dict[sample["id"]] = "Wrong"

    if len(samples) == 0:
        return {"acc": 0}
    return judge_dict, {"acc": pred_correct / len(samples)}


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10
    """
    for char in [",", ".", "!", "?", ";", ":", "'", "、", "。", "！", "？", "；", "："]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match
    japanese_char_pattern = r"[\u3040-\u30FF\u4E00-\u9FFF]"
    index_ans = True
    ans_with_brack = False
    candidates = []
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

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


def extract_numbers(string):
    """
    Exact all forms of numbers from a string with regex.
    """
    # Pattern for numbers with commas
    pattern_commas = r"-?\b\d{1,3}(?:,\d{3})+\b"
    # Pattern for scientific notation
    pattern_scientific = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
    # pattern_simple = r'-?\d+(?:\.\d+)?(?![eE][+-]?\d+)'
    pattern_simple = r"-?(?:\d+\.\d+|\.\d+|\d+)(?![eE][+-]?\d+)(?![,\d])"
    # Pattern for Japanese numbers
    pattern_japanese = r"(\d+)(?:つ|個|度|円|人|年|匹|台|%)"

    # Extract numbers with commas
    numbers_with_commas = re.findall(pattern_commas, string)
    # Extract numbers in scientific notation
    numbers_scientific = re.findall(pattern_scientific, string)
    # Extract simple numbers without commas
    numbers_simple = re.findall(pattern_simple, string)
    # Extract Japanese numbers
    numbers_japanese = re.findall(pattern_japanese, string)
    # Combine all extracted numbers
    all_numbers = (
        numbers_with_commas + numbers_scientific + numbers_simple + numbers_japanese
    )
    return all_numbers


def check_is_number(string):
    """
    Check if the given string a number.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L65
    """
    try:
        float(string.replace(",", ""))
        return True
    except ValueError:
        # check if there's comma inside
        return False


def normalize_str(string):
    """
    Normalize the str to lower case and make them float numbers if possible.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L76
    """
    # check if characters in the string

    # if number, numerize it.
    string = string.strip()

    is_number = check_is_number(string)

    if is_number:
        string = string.replace(",", "")
        string = float(string)
        # leave 2 decimal
        string = round(string, 2)
        return [string]
    else:  # it's likely to be a string
        # lower it
        string = string.lower()
        if len(string) == 1:
            return [" " + string, string + " "]  # avoid trivial matches
        return [string]


def parse_open_response(response):
    """
    Parse the prediction from the generated response.
    Return a list of predicted strings or numbers.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L122
    """

    # content = content.strip("\n").strip(".").strip(" ")
    def get_key_subresponses(response):
        key_responses = []
        response = response.strip().strip("。")
        sub_responses = re.split(r"[。！？.]\s*|\n", response)

        indicators_of_keys = [
            "よって",
            "よって、",
            "答えは",
            "答えは、",
            "最終的に",
            "最終的に、",
            "解答は",
            "解答は、" "回答は",
            "回答は、",
        ]
        key_responses = []
        for index, resp in enumerate(sub_responses):
            # if last one, accept it's an equation (the entire response can be just one sentence with equation)
            if index == len(sub_responses) - 1:
                indicators_of_keys.extend(["＝", "="])
            shortest_key_response = None  # the shortest response that may contain the answer (tail part of the response)
            for indicator in indicators_of_keys:
                if indicator in resp:
                    if not shortest_key_response:
                        shortest_key_response = resp.split(indicator)[-1].strip()
                    else:
                        if len(resp.split(indicator)[-1].strip()) < len(
                            shortest_key_response
                        ):
                            shortest_key_response = resp.split(indicator)[-1].strip()
                    # key_responses.append(resp.split(indicator)[1].strip())

            if shortest_key_response:
                # and it's not trivial
                if shortest_key_response.strip() not in [
                    ",",
                    ".",
                    "!",
                    "?",
                    ";",
                    ":",
                    "'",
                    "、",
                    "。",
                    "！",
                    "？",
                    "；",
                    "：",
                ]:
                    key_responses.append(shortest_key_response)
        if len(key_responses) == 0:  # did not found any
            return [response]
        return key_responses

    # pdb.set_trace()
    key_responses = get_key_subresponses(response)

    pred_list = key_responses.copy()  # keep the original string response
    for resp in key_responses:
        pred_list.extend(extract_numbers(resp))

    tmp_pred_list = []
    for i in range(len(pred_list)):
        tmp_pred_list.extend(normalize_str(pred_list[i]))
    pred_list = tmp_pred_list

    # remove duplicates
    pred_list = list(set(pred_list))

    return pred_list


def get_multi_choice_info(options):
    """
    Given the list of options for multiple choice question
    Return the index2ans and all_choices
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/data_utils.py#L54
    """

    start_chr = "A"
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))

    return index2ans, all_choices


@register_task("jmmmu")
class JMMMU(Task):
    def __init__(self, config=None) -> None:
        super().__init__(config)

    @property
    def dataset(self):
        """Returns the dataset associated with this class."""
        return self._dataset

    def prepare_task(self, config) -> None:
        self._dataset = self.prepare_dataset(config)
        self._dataset = self._dataset.map(
            lambda x: {
                "input_text": jmmmu_doc_to_text(x),
                "question_id": x["id"],
                "answer": x["answer"],
            }
        )
        # image is not instanciated as the list of images cannot contained in the dataset

    def prepare_dataset(self, config) -> Dataset:
        # データセットをロード
        SUBJECTS = [
            "Accounting",
            "Agriculture",
            "Architecture_and_Engineering",
            "Basic_Medical_Science",
            "Biology",
            "Chemistry",
            "Clinical_Medicine",
            "Computer_Science",
            "Design",
            "Diagnostics_and_Laboratory_Medicine",
            "Economics",
            "Electronics",
            "Energy_and_Power",
            "Finance",
            "Japanese_Art",
            "Japanese_Heritage",
            "Japanese_History",
            "Manage",
            "Marketing",
            "Materials",
            "Math",
            "Mechanical_Engineering",
            "Music",
            "Pharmacy",
            "Physics",
            "Psychology",
            "Public_Health",
            "World_History",
        ]
        dataset = None
        for subject in SUBJECTS:
            if dataset is None:
                dataset = load_dataset("JMMMU/JMMMU", name=subject, split="test")
            else:
                dataset = concatenate_datasets(
                    [dataset, load_dataset("JMMMU/JMMMU", name=subject, split="test")]
                )
        return dataset

    def doc_to_text(self, doc):
        return doc["input_text"]

    def doc_to_visual(self, doc):
        return jmmmu_doc_to_visual(doc)

    def doc_to_id(self, doc):
        return doc["question_id"]

    def process_pred(self, doc, pred):
        processed = doc
        processed["pred"] = pred
        pred = pred["text"]
        if doc["question_type"] == "multiple-choice":
            index2ans, all_choices = get_multi_choice_info(
                ast.literal_eval(doc["options"])
            )
            parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)
        else:
            parsed_pred = parse_open_response(pred)
        id = doc["question_id"]
        jmmmu_acc = {
            "id": id,
            "subdomain": extract_subset_name(doc["question_id"]),
            "question_type": doc["question_type"],
            "answer": doc["answer"],
            "parsed_pred": parsed_pred,
        }
        return jmmmu_acc

    def evaluate(self, docs: list, preds: list) -> list[dict]:
        """Evaluate batch prediction.
        Args:
        doc : list of instance of the eval dataset
        pred : list of dict with keys: { 'question_id', 'text' }
        Returns:
        eval_results: list of dictionary with keys:
            { 'input_text', 'pred', 'qa_id','answer', 'score' }
        """
        pass

    def compute_metrics(self, preds, model_id="gpt-4o-mini-2024-07-18", batch_size=10):
        """Process the results of the model.
        Args:
            jsonl_path: jsonl_path
            preds: [pred]
            model_id: openai api's model name (default: "gpt-4o-mini-2024-07-18")
        Return:
            metrics: a dictionary with key: { 'rouge-l' : rouge-l score }
            eval_results: a list of dictionaries with keys:
                { 'input_text', 'pred', 'qa_id','answer', 'score' }
        """
        eval_results = []
        docs = self.dataset

        results = [self.process_pred(doc, pred) for doc, pred in zip(docs, preds)]
        eval_results = [
            {
                "input_text": doc["input_text"],
                "pred": pred,
                "qa_id": doc["question_id"],
                "answer": doc["answer"],
                "score": int(result["answer"] == result["parsed_pred"]),
            }
            for doc, pred, result in zip(docs, preds, results)
        ]

        metrics = {"jmmmu": jmmmu_aggregate_results(results)}
        # metrics = {"jmmmu": {"Overall-Art and Psychology": {"num": 90, "acc": 0.22222}
        # convert metrics to {"Overall-Art and Psychology": 0.22222, ...}
        metrics = {
            key: value["acc"]
            for key, value in metrics["jmmmu"].items()
        }
        return metrics, eval_results

    def format_result(self, preds: list[dict], eval_results: list[dict]) -> list[dict]:
        """Format the result of the model.
        Args:
            preds:
                list of dictionaries with keys:
                {
                    "question_id": str, "text": str,
                }
            eval_results:
                list of dictionaries with keys:
                {
                    "score", # etc.
                }
        Return:
            dictonaries with keys:
            {
                    "question_id": str,
                    "text": str,
                    "score": float,
            }
        """
        assert len(preds) == len(
            eval_results
        ), "Length of preds and eval_results must be equal."
        results = []
        for pred, eval_result in zip(preds, eval_results):
            result = {
                "question_id": pred["question_id"],
                "text": pred["text"],
                "score": eval_result["score"],
            }
            results.append(result)
        return results
