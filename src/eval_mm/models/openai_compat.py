"""VLM client for OpenAI-compatible chat completion servers.

One client covers models served locally with ``vllm serve`` or SGLang and
models behind the OpenAI API, so adding a model needs no per-model wrapper.
"""

from __future__ import annotations

import base64
import io
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Literal

from loguru import logger
from PIL import Image

from ..utils.reasoning import split_reasoning
from .base_vlm import BaseVLM
from .generation_config import GenerationConfig

Thinking = Literal["on", "off"]


@dataclass
class Generation:
    """One model response, with reasoning kept apart from the answer."""

    text: str
    reasoning: str | None = None
    finish_reason: str | None = None
    completion_tokens: int | None = None
    error: str | None = None


def encode_image(image: Image.Image, image_format: str = "PNG") -> str:
    """Return *image* as a ``data:`` URL."""
    if image_format == "JPEG" and image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{image_format.lower()};base64,{encoded}"


class OpenAICompatVLM(BaseVLM):
    """Generate answers through ``/v1/chat/completions``.

    Parameters
    ----------
    model_id
        Name used for results. Also sent as the request's ``model`` unless
        *served_model_name* is given.
    base_url
        Server URL such as ``http://localhost:8000/v1``. ``None`` uses the
        OpenAI API.
    api_key
        API key. Defaults to ``OPENAI_API_KEY``; local servers accept any value.
    thinking
        ``"on"`` / ``"off"`` sets ``chat_template_kwargs.enable_thinking`` for
        servers that apply a chat template (vLLM, SGLang). ``None`` leaves the
        model's default.
    reasoning_effort
        Passed as ``reasoning_effort`` for APIs that take it.
    send_sampling_params
        Set to ``False`` for APIs that reject ``temperature`` / ``top_p``.
    """

    def __init__(
        self,
        model_id: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        served_model_name: str | None = None,
        concurrency: int = 16,
        thinking: Thinking | None = None,
        reasoning_effort: str | None = None,
        send_sampling_params: bool = True,
        extra_body: dict[str, Any] | None = None,
        image_format: str = "PNG",
        timeout: float = 600.0,
        max_retries: int = 5,
        client: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.base_url = base_url
        self.served_model_name = served_model_name or model_id
        self.concurrency = max(1, concurrency)
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.send_sampling_params = send_sampling_params
        self.extra_body = dict(extra_body or {})
        self.image_format = image_format

        if client is None:
            from openai import OpenAI

            if api_key is None:
                api_key = os.getenv("OPENAI_API_KEY") or ("EMPTY" if base_url else None)
            client = OpenAI(
                base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries
            )
        self.client = client

    def describe(self) -> dict[str, Any]:
        """Settings that affect outputs, for the run manifest."""
        return {
            "backend": "openai",
            "base_url": self.base_url,
            "served_model_name": self.served_model_name,
            "thinking": self.thinking,
            "reasoning_effort": self.reasoning_effort,
            "send_sampling_params": self.send_sampling_params,
            "extra_body": self.extra_body,
            "image_format": self.image_format,
        }

    def _request_kwargs(
        self, images: list[Image.Image] | None, text: str, gen_kwargs: GenerationConfig
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": encode_image(img, self.image_format)}}
            for img in images or []
        ]
        content.append({"type": "text", "text": text})

        kwargs: dict[str, Any] = {
            "model": self.served_model_name,
            "messages": [{"role": "user", "content": content}],
            "max_completion_tokens": gen_kwargs.max_new_tokens,
        }
        if self.send_sampling_params:
            kwargs["temperature"] = gen_kwargs.temperature
            kwargs["top_p"] = gen_kwargs.top_p
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort

        extra_body = dict(self.extra_body)
        if self.thinking is not None:
            template_kwargs = dict(extra_body.get("chat_template_kwargs", {}))
            template_kwargs["enable_thinking"] = self.thinking == "on"
            extra_body["chat_template_kwargs"] = template_kwargs
        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs

    def _generate_one(
        self, images: list[Image.Image] | None, text: str, gen_kwargs: GenerationConfig
    ) -> Generation:
        try:
            response = self.client.chat.completions.create(
                **self._request_kwargs(images, text, gen_kwargs)
            )
        except Exception as e:  # noqa: BLE001 - one failed sample must not stop the run
            logger.error(f"Request failed: {e}")
            return Generation(text="", error=f"{type(e).__name__}: {e}")

        choice = response.choices[0]
        message = choice.message
        # Servers with a reasoning parser return reasoning in its own field;
        # the field name differs between server versions.
        reasoning = getattr(message, "reasoning_content", None) or getattr(
            message, "reasoning", None
        )
        text_out = message.content or ""
        if reasoning is None:
            text_out, reasoning = split_reasoning(text_out)
        usage = getattr(response, "usage", None)
        return Generation(
            text=text_out.strip(),
            reasoning=reasoning,
            finish_reason=choice.finish_reason,
            completion_tokens=getattr(usage, "completion_tokens", None),
        )

    def batch_generate_detailed(
        self,
        images_list: list[list[Image.Image]] | None,
        text_list: list[str],
        gen_kwargs: GenerationConfig = GenerationConfig(),
    ) -> list[Generation]:
        if images_list is None:
            images_list = [[] for _ in text_list]
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(
                pool.map(
                    lambda pair: self._generate_one(pair[0], pair[1], gen_kwargs),
                    zip(images_list, text_list),
                )
            )

    def batch_generate(
        self,
        images_list: list[list[Image.Image]] | None,
        text_list: list[str],
        gen_kwargs: GenerationConfig = GenerationConfig(),
    ) -> list[str]:
        return [g.text for g in self.batch_generate_detailed(images_list, text_list, gen_kwargs)]

    def generate(
        self,
        images: list[Image.Image] | None,
        text: str,
        gen_kwargs: GenerationConfig = GenerationConfig(),
    ) -> str:
        generation = self._generate_one(images, text, gen_kwargs)
        if generation.error:
            raise RuntimeError(generation.error)
        return generation.text


def generation_to_record(generation: Generation) -> dict[str, Any]:
    """Prediction fields for *generation*, omitting the empty ones."""
    return {k: v for k, v in asdict(generation).items() if v is not None}
