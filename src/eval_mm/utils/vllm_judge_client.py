"""Offline vLLM-backed judge client.

Drop-in replacement for :class:`eval_mm.utils.azure_client.OpenAIChatAPI`
that runs the judge model locally via ``vllm.LLM`` instead of hitting the
OpenAI API. Useful for cheap / fully-local scoring with open-weight
models like ``openai/gpt-oss-20b``.

Only the ``batch_generate_chat_response`` method is implemented — the rest
of the OpenAIChatAPI surface is not used by the scorers.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class VLLMChatAPI:
    def __init__(
        self,
        model_id: str = "openai/gpt-oss-20b",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        dtype: str = "auto",
        reasoning_effort: str | None = "low",
        extra_llm_kwargs: dict[str, Any] | None = None,
    ) -> None:
        from vllm import LLM

        self.model_id = model_id
        self.reasoning_effort = reasoning_effort

        llm_kwargs: dict[str, Any] = {
            "model": model_id,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        if extra_llm_kwargs:
            llm_kwargs.update(extra_llm_kwargs)

        logger.info(f"Loading vLLM judge: {model_id} ({llm_kwargs})")
        self.llm = LLM(**llm_kwargs)

    def batch_generate_chat_response(
        self,
        chat_messages_list: list[list[dict[str, str]]],
        model_name: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        seed: int = 0,
        top_p: float = 1.0,
        stop: str | list[str] | None = None,
        **_: Any,
    ) -> list[str]:
        """Return generated text for each chat conversation.

        ``model_name`` is ignored (the embedded ``LLM`` is fixed at init time);
        accepted for interface parity with :class:`OpenAIChatAPI`.
        """
        from vllm import SamplingParams

        sp = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            stop=stop,
        )

        chat_kwargs: dict[str, Any] = {
            "messages": chat_messages_list,
            "sampling_params": sp,
            "use_tqdm": False,
        }
        if self.reasoning_effort is not None:
            chat_kwargs["chat_template_kwargs"] = {
                "reasoning_effort": self.reasoning_effort,
            }

        outputs = self.llm.chat(**chat_kwargs)
        return [o.outputs[0].text if o.outputs else "" for o in outputs]

    def __repr__(self) -> str:
        return f"VLLMChatAPI(model_id={self.model_id!r})"
