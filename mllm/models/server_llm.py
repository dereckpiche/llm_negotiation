from concurrent.futures import ThreadPoolExecutor
from typing import List

import backoff
from openai import OpenAI, RateLimitError

from mllm.utils.common_imports import *


class ServerLLM:
    """
    ServerLLM is an agent that utilizes the OpenAI API for generating responses in a conversational manner.
    It supports prompting the model and managing conversation history.
    """

    def __init__(
        self,
        name: str = "openai_agent",
        api_key: str = "",
        model: str = "gpt-4o",  # default OpenAI model
    ) -> None:
        """
        Initializes the ServerLLM.

        Args:
            name (str): The name of the agent.
            api_key (str): The API key for accessing OpenAI's API.
            model (str): The model to be used, specified by the model name (default is 'gpt-3.5-turbo').
            out_folder (str): The output folder for saving conversation history and logs.
        """
        self.name = name
        self.api_key = api_key
        if os.getenv("OPENAI_API_KEY"):
            self.client = OpenAI()
        else:
            self.client = OpenAI(api_key=api_key)
        self.model = model

    @backoff.on_exception(backoff.expo, RateLimitError)
    def _sync_fetch(self, prompt: dict) -> str:
        response = self.client.chat.completions.create(
            model=self.model, messages=prompt, max_tokens=150
        )
        return response.choices[0].message.content.strip()

    def prompt(self, contexts: List[dict]) -> str:
        """
        Generates a response from the OpenAI model based on the provided contexts.

        Args:
            contexts (List[dict]): The contexts for generation.

        scores:
            str: The generated response from the model.
        """
        if not contexts:
            return ""

        with ThreadPoolExecutor(max_workers=min(64, len(contexts))) as executor:
            responses = list(executor.map(self._sync_fetch, contexts))

        return responses

    def set_adapter(self, name: str) -> None:
        """
        Dummy method for setting an adapter. Does nothing.

        Args:
            name (str): The name of the adapter to switch to.
        """
        pass
