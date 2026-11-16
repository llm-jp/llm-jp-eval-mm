"""Tests for the OpenAI-compatible backend. No network or GPU needed."""

import json
from types import SimpleNamespace

import pytest
from PIL import Image

from eval_mm.models.generation_config import GenerationConfig
from eval_mm.models.openai_compat import OpenAICompatVLM, encode_image
from eval_mm.result_schema import RunManifest, load_manifest, write_manifest
from eval_mm.utils.azure_client import adapt_rejected_params
from eval_mm.utils.reasoning import split_reasoning


class FakeClient:
    """Stands in for ``openai.OpenAI`` and records the requests it receives."""

    def __init__(self, respond):
        self.requests: list[dict] = []
        self._respond = respond
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self._respond(kwargs)


def completion(content, finish_reason="stop", **message_fields):
    message = SimpleNamespace(content=content, **message_fields)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=SimpleNamespace(completion_tokens=7),
    )


@pytest.mark.parametrize(
    ("text", "answer", "reasoning"),
    [
        ("B", "B", None),
        ("<think>A か B か</think>\n\nB", "B", "A か B か"),
        ("A か B か</think>B", "B", "A か B か"),
        ("<think>まだ考えている", "", "まだ考えている"),
        ("", "", None),
        (None, "", None),
    ],
)
def test_split_reasoning(text, answer, reasoning):
    assert split_reasoning(text) == (answer, reasoning)


def test_request_carries_images_and_generation_settings():
    client = FakeClient(lambda _: completion("犬"))
    model = OpenAICompatVLM("org/model", client=client, served_model_name="served")
    image = Image.new("RGB", (4, 4), "red")

    out = model.generate([image, image], "何が写っていますか", GenerationConfig(max_new_tokens=32))

    assert out == "犬"
    request = client.requests[0]
    assert request["model"] == "served"
    assert request["max_completion_tokens"] == 32
    assert request["temperature"] == 0.0
    content = request["messages"][0]["content"]
    assert [part["type"] for part in content] == ["image_url", "image_url", "text"]
    assert content[0]["image_url"]["url"] == encode_image(image)
    assert content[-1]["text"] == "何が写っていますか"
    assert "extra_body" not in request


@pytest.mark.parametrize(("thinking", "expected"), [("on", True), ("off", False)])
def test_thinking_sets_chat_template_kwargs(thinking, expected):
    client = FakeClient(lambda _: completion("ok"))
    model = OpenAICompatVLM("m", client=client, thinking=thinking)
    model.generate(None, "q")
    assert client.requests[0]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": expected}
    }


def test_sampling_params_can_be_omitted():
    client = FakeClient(lambda _: completion("ok"))
    model = OpenAICompatVLM(
        "m", client=client, send_sampling_params=False, reasoning_effort="low"
    )
    model.generate(None, "q")
    request = client.requests[0]
    assert "temperature" not in request and "top_p" not in request
    assert request["reasoning_effort"] == "low"


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
def test_reasoning_field_from_server_is_kept_apart(field):
    client = FakeClient(lambda _: completion("B", **{field: "考えた"}))
    generation = OpenAICompatVLM("m", client=client).batch_generate_detailed(None, ["q"])[0]
    assert (generation.text, generation.reasoning) == ("B", "考えた")
    assert generation.completion_tokens == 7


def test_think_tags_are_split_when_server_has_no_parser():
    client = FakeClient(lambda _: completion("<think>考えた</think>B"))
    generation = OpenAICompatVLM("m", client=client).batch_generate_detailed(None, ["q"])[0]
    assert (generation.text, generation.reasoning) == ("B", "考えた")


def test_truncated_reasoning_gives_empty_answer_with_finish_reason():
    client = FakeClient(lambda _: completion(None, finish_reason="length", reasoning="長い"))
    generation = OpenAICompatVLM("m", client=client).batch_generate_detailed(None, ["q"])[0]
    assert generation.text == ""
    assert generation.finish_reason == "length"


def test_thinking_cut_off_before_the_answer_is_not_scored_as_an_answer():
    client = FakeClient(lambda _: completion("Thinking Process: 1. Analyze", finish_reason="length"))
    on = OpenAICompatVLM("m", client=client, thinking="on").batch_generate_detailed(None, ["q"])[0]
    assert (on.text, on.reasoning) == ("", "Thinking Process: 1. Analyze")

    # Without thinking, truncated output is still the model's answer.
    off = OpenAICompatVLM("m", client=client, thinking="off").batch_generate_detailed(None, ["q"])[0]
    assert off.text == "Thinking Process: 1. Analyze"


def test_failed_request_is_recorded_and_order_is_kept():
    def respond(kwargs):
        text = kwargs["messages"][0]["content"][-1]["text"]
        if text == "bad":
            raise RuntimeError("boom")
        return completion(text.upper())

    model = OpenAICompatVLM("m", client=FakeClient(respond), concurrency=4)
    generations = model.batch_generate_detailed(None, ["a", "bad", "c"])

    assert [g.text for g in generations] == ["A", "", "C"]
    assert "boom" in generations[1].error
    with pytest.raises(RuntimeError):
        model.generate(None, "bad")


def test_adapt_rejected_params():
    kwargs = {"max_tokens": 256, "temperature": 0.0, "seed": 0}
    step1 = adapt_rejected_params(
        kwargs, "Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens' instead."
    )
    assert step1 == {"max_completion_tokens": 256, "temperature": 0.0, "seed": 0}
    step2 = adapt_rejected_params(step1, "Unsupported value: 'temperature' does not support 0.0")
    assert step2 == {"max_completion_tokens": 256, "seed": 0}
    assert adapt_rejected_params(step2, "Invalid image") is None
    assert kwargs == {"max_tokens": 256, "temperature": 0.0, "seed": 0}


def test_manifest_keeps_generation_info_when_scored_later(tmp_path):
    out = str(tmp_path)
    write_manifest(
        out, "m", "t", [],
        generation={"config": {"max_new_tokens": 32}, "model": {"thinking": "off"}},
        dataset={"num_samples": 3},
    )
    write_manifest(out, "m", "t", ["exact-match"], judge_model="judge")

    manifest = load_manifest(out)
    assert manifest.metrics == ["exact-match"]
    assert manifest.judge_model == "judge"
    assert manifest.generation["model"]["thinking"] == "off"
    assert manifest.dataset == {"num_samples": 3}
    assert "packages" in manifest.environment


def test_manifest_from_older_or_newer_schema_loads():
    old = json.dumps({"schema_version": "1.0", "model_id": "m", "task_id": "t"})
    assert RunManifest.from_json(old).generation == {}
    newer = json.dumps({"model_id": "m", "field_added_later": 1})
    assert RunManifest.from_json(newer).model_id == "m"
