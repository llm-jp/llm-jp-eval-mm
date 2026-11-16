import os
import asyncio
import logging
from openai import AsyncAzureOpenAI, AsyncOpenAI, APIError, BadRequestError
from openai.types.chat import ChatCompletion
import backoff

from ..metadata import DEFAULT_JUDGE_MODEL
from tqdm import tqdm

logger = logging.getLogger(__name__)


def adapt_rejected_params(kwargs: dict, error_message: str) -> dict | None:
    """Rewrite *kwargs* after a 400 that names a parameter the model rejects.

    Newer models take ``max_completion_tokens`` instead of ``max_tokens`` and
    some accept only the default sampling parameters. Returns ``None`` when
    the error is about something else.
    """
    message = error_message.lower()
    adapted = dict(kwargs)
    if "max_tokens" in message and "max_tokens" in adapted:
        adapted["max_completion_tokens"] = adapted.pop("max_tokens")
        return adapted
    for name in ("temperature", "top_p", "seed"):
        if name in message and name in adapted:
            del adapted[name]
            return adapted
    return None


@backoff.on_exception(
    backoff.expo,
    APIError,
    max_tries=5,
    giveup=lambda e: isinstance(e, BadRequestError),
)
async def _create_completion(
    client: AsyncOpenAI,
    model_name: str,
    messages_list: list[dict[str, str]],
    **kwargs,
) -> ChatCompletion:
    return await client.chat.completions.create(
        model=model_name,
        messages=messages_list,
        **kwargs,
    )


async def call_openai(
    client: AsyncOpenAI,
    model_name: str,
    messages_list: list[dict[str, str]],
    **kwargs,
) -> ChatCompletion:
    while True:
        try:
            return await _create_completion(client, model_name, messages_list, **kwargs)
        except BadRequestError as e:
            adapted = adapt_rejected_params(kwargs, str(e))
            if adapted is None:
                raise
            kwargs = adapted


class OpenAIChatAPI:
    def __init__(self) -> None:
        if os.getenv("AZURE_OPENAI_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
            # Azure API version can be overridden via the AZURE_API_VERSION
            # environment variable.  The default ("2023-05-15") is the GA
            # release that supports Chat Completions; bump this when a newer
            # stable version is needed.
            api_version = os.getenv("AZURE_API_VERSION", "2023-05-15")
            self.client = AsyncAzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_KEY"),
                api_version=api_version,
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            )
        elif os.getenv("OPENAI_API_KEY"):
            self.client = AsyncOpenAI()
        else:
            raise ValueError(
                "API Key not found. Please set the OPENAI_API_KEY or AZURE_OPENAI_KEY environment variables."
            )

    async def _async_batch_run_chatgpt(
        self,
        messages_list: list[list[dict[str, str]]],
        stop_sequences: str | list[str] | None = None,
        max_new_tokens: int | None = None,
        model_name: str | None = None,
        **kwargs,
    ) -> list[ChatCompletion | Exception]:
        if stop_sequences is not None:
            if "stop" in kwargs:
                raise ValueError("Specify only one: `stop_sequences` or `stop`.")
            kwargs["stop"] = stop_sequences

        if max_new_tokens is not None:
            if "max_tokens" in kwargs:
                raise ValueError("Specify only one: `max_new_tokens` or `max_tokens`.")
            kwargs["max_tokens"] = max_new_tokens

        tasks = [
            asyncio.create_task(call_openai(self.client, model_name, ms, **kwargs))
            for ms in messages_list
        ]
        results: list[ChatCompletion | Exception] = await asyncio.gather(
            *tasks, return_exceptions=True
        )

        output: list[ChatCompletion | Exception] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error in task {i}: {result}")
                output.append(None)
            else:
                output.append(result)
        return output

    def batch_generate_chat_response(
        self,
        chat_messages_list: list[list[dict[str, str]]],
        model_name: str | None = None,
        **kwargs,
    ) -> list[str]:
        api_responses = asyncio.run(
            self._async_batch_run_chatgpt(
                chat_messages_list, model_name=model_name, **kwargs
            )
        )
        model_outputs = []
        for res in api_responses:
            if isinstance(res, ChatCompletion):
                model_output = res.choices[0].message.content
                model_outputs.append(model_output)

                logger.info(f"Model output: {model_output}")
                logger.info(f"Usage: {res.usage}")
            else:
                logger.error(f"Unexpected error: {res}")
                model_outputs.append("")
        return model_outputs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(client={self.client})"


class MockChatAPI:
    def __init__(self) -> None:
        pass

    def batch_generate_chat_response(
        self,
        chat_messages_list: list[list[dict[str, str]]],
        model_name: str | None = None,
        **kwargs,
    ) -> list[str]:
        return ["Mock response"] * len(chat_messages_list)


if __name__ == "__main__":
    # Test code
    client = OpenAIChatAPI()
    messages_list = [
        [{"role": "system", "content": "こんにちは"}],
        [{"role": "user", "content": "今日の天気は？"}],
    ]

    responses = client.batch_generate_chat_response(
        messages_list, model_name=DEFAULT_JUDGE_MODEL
    )
    print(responses)

    # Example dataset
    dataset = [messages_list[0] for _ in range(10)]

    # Batch processing with progress bar
    with tqdm(total=len(dataset)) as pbar:
        batch_size = 5
        for i in range(0, len(dataset), batch_size):
            items = dataset[i : i + batch_size]
            responses = client.batch_generate_chat_response(
                items, model_name=DEFAULT_JUDGE_MODEL
            )
            print(responses)
            pbar.update(len(items))
