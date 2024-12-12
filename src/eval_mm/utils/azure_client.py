import os
from openai import AsyncAzureOpenAI, AsyncOpenAI
import asyncio
from typing import Optional
import openai
import logging
import backoff
from tqdm import tqdm

logger = logging.getLogger(__name__)


@backoff.on_exception(backoff.expo, openai.APIError, max_tries=5)
async def call_openai(
    client: AsyncOpenAI, model_name: str, messages_list: list[dict[str, str]], **kwargs
) -> dict:
    """
    Calls OpenAI's chat completion API and handles retries on failure.
    """
    return await client.chat.completions.create(
        model=model_name,
        messages=messages_list,
        **kwargs,
    )


class OpenAIChatAPI:
    """
    Wrapper class for the OpenAI API client.
    Provides a method to send multiple chat requests in parallel.
    """

    def __init__(self) -> None:
        if os.getenv("AZURE_OPENAI_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
            self.client = AsyncAzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_KEY"),
                api_version="2023-05-15",
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
        stop_sequences: Optional[str | list[str]] = None,
        max_new_tokens: Optional[int] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> list[Optional[dict]]:
        """
        Send multiple chat requests to the OpenAI API in parallel.
        """
        if stop_sequences is not None:
            if "stop" in kwargs:
                raise ValueError(
                    "You specified both `stop_sequences` and `stop` in generation kwargs. Specify only one."
                )
            kwargs["stop"] = stop_sequences

        if max_new_tokens is not None:
            if "max_tokens" in kwargs:
                raise ValueError(
                    "You specified both `max_new_tokens` and `max_tokens` in generation kwargs. Specify only one."
                )
            kwargs["max_tokens"] = max_new_tokens

        tasks = [
            asyncio.create_task(call_openai(self.client, model_name, ms, **kwargs))
            for ms in messages_list
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error in task {i}: {result}")
                results[i] = None

        return results

    def batch_generate_chat_response(
        self,
        chat_messages_list: list[list[dict[str, str]]],
        model_name: Optional[str] = None,
        **kwargs,
    ) -> list[str]:
        """
        Generate chat responses for a batch of messages.
        """
        api_responses = asyncio.run(
            self._async_batch_run_chatgpt(
                chat_messages_list, model_name=model_name, **kwargs
            )
        )
        model_outputs = []
        for res in api_responses:
            if res is None:
                model_outputs.append("")
                continue
            model_output = res.choices[0].message.content
            model_outputs.append(model_output)

            logger.info(f"Model output: {model_output}")
            logger.info(f"Usage: {res.usage}")
        return model_outputs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(client={self.client})"


if __name__ == "__main__":
    # Test code
    client = OpenAIChatAPI()
    messages_list = [
        [{"role": "system", "content": "こんにちは"}],
        [{"role": "user", "content": "今日の天気は？"}],
    ]

    responses = client.batch_generate_chat_response(
        messages_list, model_name="gpt-4o-mini-2024-07-18"
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
                items, model_name="gpt-4o-mini-2024-07-18"
            )
            print(responses)
            pbar.update(len(items))
