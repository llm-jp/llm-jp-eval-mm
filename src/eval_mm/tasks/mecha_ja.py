from datasets import Dataset, load_dataset
from ..api.registry import register_task
from ..api.task import Task
from eval_mm.metrics import ScorerRegistry

MULTI_CHOICE_PROMPT = (
    "与えられた選択肢の中から最も適切な回答のアルファベットを直接記入してください。"
)

OPTIONS_MAP = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
}


def parse_options(options):
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join(
        [
            f"{option_letter}. {option}"
            for option_letter, option in zip(option_letters, options)
        ]
    )
    return choices_str


def construct_prompt(question, options):
    parsed_options = parse_options(options)
    return f"{question}\n{parsed_options}\n\n{MULTI_CHOICE_PROMPT}"


def rotate_single_example(doc):
    """
    1つの doc に対して4パターン (options を左回転0,1,2,3) を作り、
    その際に answer も回転にあわせて更新し、各パターンの辞書をリストで返す。
    """
    base_opts = doc["options"]
    n = len(base_opts)  # 4想定
    orig_answer_idx = doc["answer"]  # 0~3
    results = []
    for i in range(n):
        rotated_options = base_opts[i:] + base_opts[:i]
        new_answer_idx = (orig_answer_idx - i) % n
        new_doc = dict(doc)
        new_doc["options"] = rotated_options
        new_doc["answer"] = new_answer_idx
        new_doc["answer_text"] = OPTIONS_MAP[new_answer_idx]
        new_doc["question_id"] = f"{doc['question_id']}_rot{i}"
        new_doc["input_text"] = construct_prompt(
            new_doc["question"], new_doc["options"]
        )
        results.append(new_doc)
    return results


def rotate_options_fn(batch):
    """
    batched=True 用の関数。
    batch: dict of lists
    これを1つずつ取り出して rotate_single_example で4つに拡張し、
    最終的に「列ごとのリスト」を返す。
    """
    # 出力用の空リストを用意
    new_batch = {
        "question": [],
        "options": [],
        "answer": [],
        "answer_text": [],
        "image": [],
        "question_id": [],
        "answer_type": [],
        "background_text": [],
        "input_text": [],
    }

    num_examples = len(batch["question_id"])
    for i in range(num_examples):
        # i番目のサンプル doc をまとめる
        doc = {
            "question": batch["question"][i],
            "options": batch["options"][i],
            "answer": batch["answer"][i],
            "answer_type": batch["answer_type"][i],
            "image": batch["image"][i],
            "background_text": batch["background_text"][i],
            "question_id": batch["question_id"][i],
        }
        # rotateして複数サンプルに展開
        rotated_docs = rotate_single_example(doc)
        # new_batch にappend
        for rd in rotated_docs:
            new_batch["question"].append(rd["question"])
            new_batch["options"].append(rd["options"])
            new_batch["answer"].append(rd["answer"])
            new_batch["answer_text"].append(rd["answer_text"])
            new_batch["image"].append(rd["image"])
            new_batch["question_id"].append(rd["question_id"])
            new_batch["answer_type"].append(rd["answer_type"])
            new_batch["background_text"].append(rd["background_text"])
            new_batch["input_text"].append(rd["input_text"])

    return new_batch


@register_task("mecha-ja")
class MECHAJa(Task):
    def __init__(self, config):
        super().__init__(config)

        ds = self._prepare_dataset()

        # rotate_choices が有効なら4パターン展開
        if getattr(self.config, "rotate_choices", False):
            ds = ds.map(
                rotate_options_fn,
                batched=True,
                remove_columns=ds.column_names,
            )
        self.dataset = ds

    @staticmethod
    def _prepare_dataset() -> Dataset:
        ds = load_dataset("llm-jp/MECHA-ja", split="test")

        ds = ds.map(
            lambda x, idx: {
                "question": x["question"],
                "options": x["options"],
                "answer": x["answer"],  # 0~3
                "answer_type": x["answer_type"],
                "image": x["image"],
                "background_text": x["background_text"],
                "question_id": str(idx),
                "answer_text": OPTIONS_MAP[x["answer"]],
                "input_text": construct_prompt(x["question"], x["options"]),
            },
            with_indices=True,
        )

        return ds

    @staticmethod
    def doc_to_text(doc):
        return doc["input_text"]

    @staticmethod
    def doc_to_visual(doc):
        return doc["image"]

    @staticmethod
    def doc_to_id(doc):
        return doc["question_id"]

    @staticmethod
    def doc_to_answer(doc):
        return doc["answer_text"]

    def calc_scores(self, preds: list, metric: str) -> list:
        scorer = ScorerRegistry.get_scorer(metric)
        refs = [doc["answer_text"] for doc in self.dataset]
        pred_texts = [p["text"] for p in preds]
        kwargs = {
            "docs": self.dataset,
            "batch_size": self.config.batch_size_for_evaluation,
        }
        return scorer.score(refs, pred_texts, **kwargs)

    def gather_scores(self, scores: list[dict], metric: str) -> dict:
        scorer = ScorerRegistry.get_scorer(metric)
        return scorer.aggregate(scores, docs=self.dataset)
