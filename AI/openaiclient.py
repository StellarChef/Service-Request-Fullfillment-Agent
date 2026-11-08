from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

from config.config import env

CHAT_MODEL = "gpt-6-luna"
EMBEDDING_MODEL = "text-embedding-3-small"

T = TypeVar("T", bound=BaseModel)


class AIRefusalError(Exception):
    """Raised when the model refuses a structured-output request or returns nothing parseable."""


class AIClient:
    def __init__(self):
        self.client = OpenAI()

    def embed_text(self, text: str):
        response = self.client.embeddings.create(input=f"{text}", model=EMBEDDING_MODEL)
        return response.data[0].embedding

    def embed_many(self, texts: list):
        response = self.client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
        return [item.embedding for item in response.data]

    def chat(self, messages: list):
        response = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages
        )

        return response.choices[0].message.content

    def parse(self, messages: list, schema: type[T]) -> T:
        """Chat call with structured output: the reply is validated against `schema`."""
        response = self.client.chat.completions.parse(
            model=CHAT_MODEL,
            messages=messages,
            response_format=schema,
        )
        message = response.choices[0].message
        if message.refusal:
            raise AIRefusalError(message.refusal)
        if message.parsed is None:
            raise AIRefusalError("Model returned no parsed output")
        return message.parsed
