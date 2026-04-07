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


MULTI_CHOICE_PROMPT = (
    "\u4e0e\u3048\u3089\u308c\u305f\u9078\u629e\u80a2\u306e\u4e2d\u304b\u3089\u6700\u3082\u9069\u5207\u306a\u56de\u7b54\u306e\u30a2\u30eb\u30d5\u30a1\u30d9\u30c3\u30c8\u3092\u76f4\u63a5\u8a18\u5165\u3057\u3066\u304f\u3060\u3055\u3044\u3002"
)
OPEN_ENDED_PROMPT = "\u8cea\u554f\u306b\u5bfe\u3059\u308b\u56de\u7b54\u3092\u5358\u8a9e\u3084\u77ed\u3044\u30d5\u30ec\u30fc\u30ba\u3067\u8a18\u5165\u3057\u3066\u304f\u3060\u3055\u3044\u3002"

# Japanese-specific key sub-response indicators for parse_open_response
_JAPANESE_INDICATORS = [
    "\u3088\u3063\u3066",
    "\u3088\u3063\u3066\u3001",
    "\u7b54\u3048\u306f",
    "\u7b54\u3048\u306f\u3001",
    "\u6700\u7d42\u7684\u306b",
    "\u6700\u7d42\u7684\u306b\u3001",
    "\u89e3\u7b54\u306f",
    "\u89e3\u7b54\u306f\u3001"
    "\u56de\u7b54\u306f",
    "\u56de\u7b54\u306f\u3001",
]


def extract_numbers(string):
    """Exact all forms of numbers from a string with regex."""
    # Pattern for numbers with commas
    pattern_commas = r"-?\b\d{1,3}(?:,\d{3})+\b"
    # Pattern for scientific notation
    pattern_scientific = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
    pattern_simple = r"-?(?:\d+\.\d+|\.\d+|\d+)(?![eE][+-]?\d+)(?![,\d])"
    # Pattern for Japanese numbers
    pattern_japanese = r"(\d+)(?:\u3064|\u500b|\u5ea6|\u5186|\u4eba|\u5e74|\u5339|\u53f0|%)"

    numbers_with_commas = re.findall(pattern_commas, string)
    numbers_scientific = re.findall(pattern_scientific, string)
    numbers_simple = re.findall(pattern_simple, string)
    numbers_japanese = re.findall(pattern_japanese, string)
    all_numbers = (
        numbers_with_commas + numbers_scientific + numbers_simple + numbers_japanese
    )
    return all_numbers


def jmmmu_doc_to_text(doc):
    question = construct_prompt(doc, MULTI_CHOICE_PROMPT, OPEN_ENDED_PROMPT)
    # replace_images_tokens normalises numbered image placeholders (e.g.
    # "<image 1>", "<image 2>") into a single "<image>" token.  This is
    # necessary because the JMMMU dataset embeds numbered image references in
    # questions, but the VLM inference pipeline expects a uniform "<image>"
    # placeholder.
    question = replace_images_tokens(question)
    return question


@register_scorer("jmmmu")
class JMMMUScorer(Scorer):
    def score(self, refs: list[str], preds: list[str]) -> list[int]:
        docs = self.config.docs
        assert docs is not None
        scores = []
        for doc, pred in zip(docs, preds):
            score = get_score(
                doc, pred, self.config.random_choice, extract_numbers,
                open_response_indicators=_JAPANESE_INDICATORS,
            )
            scores.append(score)
        return scores

    def aggregate(self, scores: list[int]) -> AggregateOutput:
        docs = self.config.docs
        assert docs is not None
        return aggregate_mmmu_results(docs, scores, DOMAIN_CAT2SUB_CAT)
