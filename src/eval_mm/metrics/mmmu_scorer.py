import re

from .scorer import Scorer, AggregateOutput
from .scorer_registry import register_scorer
from .mmmu_utils import (
    replace_images_tokens,
    construct_prompt,
    get_score,
    aggregate_mmmu_results,
)

DOMAIN_CAT2SUB_CAT = {
    "Art and Design": ["Art", "Art_Theory", "Design", "Music"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science": [
        "Biology",
        "Chemistry",
        "Geography",
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
    "Humanities and Social Science": [
        "History",
        "Literature",
        "Sociology",
        "Psychology",
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


MULTI_CHOICE_PROMPT = "Answer with the option's letter from the given choices directly."
OPEN_ENDED_PROMPT = "Answer the question using a single word or phrase."


def extract_numbers(string):
    """Exact all forms of numbers from a string with regex.

    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L100
    """
    # Pattern for numbers with commas
    pattern_commas = r"-?\b\d{1,3}(?:,\d{3})+\b"
    # Pattern for scientific notation
    pattern_scientific = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
    # Pattern for simple numbers without commas
    pattern_simple = r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])"

    numbers_with_commas = re.findall(pattern_commas, string)
    numbers_scientific = re.findall(pattern_scientific, string)
    numbers_simple = re.findall(pattern_simple, string)

    all_numbers = numbers_with_commas + numbers_scientific + numbers_simple
    return all_numbers


def mmmu_doc_to_text(doc):
    question = construct_prompt(doc, MULTI_CHOICE_PROMPT, OPEN_ENDED_PROMPT)
    # replace_images_tokens normalises numbered image placeholders (e.g.
    # "<image 1>", "<image 2>") into a single "<image>" token.  This is
    # necessary because the MMMU dataset embeds numbered image references in
    # questions, but the VLM inference pipeline expects a uniform "<image>"
    # placeholder.
    question = replace_images_tokens(question)
    return question


@register_scorer("mmmu")
class MMMUScorer(Scorer):
    def score(self, refs: list[str], preds: list[str]) -> list[int]:
        docs = self.config.docs
        assert docs is not None
        scores = []
        for doc, pred in zip(docs, preds):
            score = get_score(doc, pred, self.config.random_choice, extract_numbers)
            scores.append(score)
        return scores

    def aggregate(self, scores: list[int]) -> AggregateOutput:
        docs = self.config.docs
        assert docs is not None
        return aggregate_mmmu_results(docs, scores, DOMAIN_CAT2SUB_CAT)
