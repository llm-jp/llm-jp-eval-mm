from eval_mm.utils.azure_client import OpenAIChatAPI
from collections import defaultdict
import numpy as np

RULES: dict = {
    "coding": {
        "role": "Assistant",
        "prompt": "Your task is to evaluate the coding abilities of the above two assistants. They have been asked to implement a program to solve a given problem. Please review their code submissions, paying close attention to their problem-solving approach, code structure, readability, and the inclusion of helpful comments.\n\nPlease ensure that the assistants' submissions:\n\n1. Correctly implement the given problem statement.\n2. Contain accurate and efficient code.\n3. Include clear and concise comments that explain the code's logic and functionality.\n4. Adhere to proper coding standards and best practices.\n\nOnce you have carefully reviewed both submissions, provide detailed feedback on their strengths and weaknesses, along with any suggestions for improvement. You should first output a single line containing two scores on the scale of 1-10 (1: no code/no sense; 10: perfect) for Assistant 1 and 2, respectively. Then give extra comments starting from the next line.",
    },
    "math": {
        "role": "Assistant",
        "prompt": "We would like to request your feedback on the mathematical proficiency of two AI assistants regarding the given user question.\nFirstly, please solve the problem independently, without referring to the answers provided by Assistant 1 and Assistant 2.\nAfterward, please examine the problem-solving process of Assistant 1 and Assistant 2 step-by-step to ensure their correctness, identifying any incorrect steps if present. Your evaluation should take into account not only the answer but also the problem-solving steps.\nFinally, please output a Python tuple containing two numerical scores for Assistant 1 and Assistant 2, ranging from 1 to 10, respectively. If applicable, explain the reasons for any variations in their scores and determine which assistant performed better.",
    },
    "default": {
        "role": "Assistant",
        "prompt": "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment. Sentences are given in Japanese, and language is not relevant to score.",
    },
    "conv": {
        "role": "Assistant",
        "prompt": "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with five descriptive sentences describing the same image and the bounding box coordinates of each object in the scene. These coordinates are in the form of bounding boxes, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment. Sentences are given in Japanese, and language is not relevant to score.",
    },
    "detail": {
        "role": "Assistant",
        "prompt": "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with five descriptive sentences describing the same image and the bounding box coordinates of each object in the scene. These coordinates are in the form of bounding boxes, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment. Sentences are given in Japanese, and language is not relevant to score.",
    },
    "complex": {
        "role": "Assistant",
        "prompt": "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with five descriptive sentences describing the same image and the bounding box coordinates of each object in the scene. These coordinates are in the form of bounding boxes, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment. Sentences are given in Japanese, and language is not relevant to score.",
    },
    "llava_bench_conv": {
        "role": "Assistant",
        "prompt": "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with a few sentences describing the image. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.",
    },
    "llava_bench_detail": {
        "role": "Assistant",
        "prompt": "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with a few sentences describing the image. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.",
    },
    "llava_bench_complex": {
        "role": "Assistant",
        "prompt": "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with a few sentences describing the image. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.",
    },
}


def parse_score(review: str) -> dict[str, int]:
    try:
        score_pair = review.split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        sp = score_pair.split(" ")
        if len(sp) == 2:
            return {"score": int(sp[1]), "score_gpt": int(sp[0])}
        else:
            print("error", review)
            return {"score": -1, "score_gpt": -1}
    except Exception as e:
        print(e)
        print("error", review)
        return {"score": -1, "score_gpt": -1}


def ask_gpt4_batch(
    content_list: str, max_tokens: int, async_client: OpenAIChatAPI, model_name: str
) -> list:
    message_list = [
        [
            {
                "role": "system",
                "content": "You are a helpful and precise assistant for checking the quality of the answer.",
            },
            {"role": "user", "content": content},
        ]
        for content in content_list
    ]
    completions = async_client.batch_generate_chat_response(
        message_list,
        max_tokens=max_tokens,
        temperature=0,
        seed=0,
        model_name=model_name,
    )
    return completions


def build_content(context, input_text, ref_answer, pred_answer, role, prompt):
    return (
        f"[Context]\n{context}\n\n"
        f"[Question]\n{input_text}\n\n"
        f"[{role} 1]\n{ref_answer}\n\n[End of {role} 1]\n\n"
        f"[{role} 2]\n{pred_answer}\n\n[End of {role} 2]\n\n"
        f"[System]\n{prompt}\n\n"
        f"If it is not relevant to the context, does not answer directly, or says the wrong thing, give it a low score.\n\n"
    )


class HeronBenchScorer:
    @staticmethod
    def score(refs, preds: list[str], **kwargs) -> list[dict[str, int]]:
        docs = kwargs["docs"]
        client = kwargs["client"]
        judge_model = kwargs["judge_model"]
        contents = []
        for doc, ref, pred in zip(docs, refs, preds):
            content = build_content(
                doc["context"],
                doc["input_text"],
                ref,
                pred,
                "Assistant",
                RULES[doc["category"]]["prompt"],
            )
            contents.append(content)
        completions = ask_gpt4_batch(contents, 1024, client, judge_model)
        scores = [parse_score(completion) for completion in completions]
        assert len(scores) == len(docs)
        return scores

    @staticmethod
    def aggregate(scores: list[dict[str, int]], **kwargs) -> dict[str, float]:
        docs = kwargs["docs"]
        category_list = ["conv", "detail", "complex"]
        heron_metrics = defaultdict(float)
        for category in category_list:
            score_owns = [
                score["score"]
                for score, doc in zip(scores, docs)
                if doc["category"] == category
            ]
            score_gpts = [
                score["score_gpt"]
                for score, doc in zip(scores, docs)
                if doc["category"] == category
            ]
            if len(score_owns) == 0 or np.mean(score_owns) == -1:
                continue
            avg_score = np.mean(score_owns)
            avs_score_rel = (
                100
                * np.mean(score_owns)
                / max(
                    0.01, np.mean(score_gpts)
                )  # divide by 0.01 when 0 division happens
            )
            heron_metrics[category] = avg_score
            heron_metrics[category + "_rel"] = avs_score_rel
        heron_metrics["parse_error_count"] = sum(
            score["score"] == -1 for score in scores
        )
        heron_metrics["overall"] = sum([score["score"] for score in scores]) / len(
            scores
        )
        heron_metrics["overall_rel"] = sum(
            [heron_metrics[category + "_rel"] for category in category_list]
        ) / len(category_list)
        return heron_metrics


def test_heron_bench_scorer():
    from datasets import load_dataset
    from eval_mm.utils.azure_client import MockChatAPI

    docs = load_dataset("Silviase/Japanese-Heron-Bench", split="train").select(range(1))
    docs = docs.rename_column("text", "input_text")
    pred_texts = ["This is a test."]
    client = MockChatAPI()
    judge_model = "gpt-4o-mini-2024-07-18"
    refs = [doc["answer"]["gpt-4-0125-preview"] for doc in docs]
    scores = HeronBenchScorer.score(
        refs, pred_texts, docs=docs, client=client, judge_model=judge_model
    )
    assert len(scores) == 1
    print(scores)
    calculated_metric = HeronBenchScorer.aggregate(scores, docs=docs)
    assert "overall" in calculated_metric
    print(calculated_metric)


if __name__ == "__main__":
    from datasets import load_dataset

    ds = load_dataset("Silviase/Japanese-Heron-Bench", split="train")
    ds = ds.rename_column("text", "input_text")
    # the 102th problem
    ds = ds.select(range(100, 102))
    pred_texts = [
        "画像から判断すると、制限速度は50 km/hのようです。道路標識に「50」と表示されています。",
        "画像に写っている標識によると、現在地からニセコまでは12kmです。",
    ]
    refs = [doc["answer"]["gpt-4-0125-preview"] for doc in ds]
    client = OpenAIChatAPI()
    judge_model = "gpt-4o-2024-05-13"
    scores = HeronBenchScorer.score(
        refs, pred_texts, docs=ds, client=client, judge_model=judge_model
    )
    print(scores)
