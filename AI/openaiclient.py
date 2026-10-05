from openai import OpenAI

from config.config import env

class AIClient:
    def __init__(self):
        self.client = OpenAI()

    def embed_text(self, text: str):
        response = self.client.embeddings.create(input=f"{text}", model="text-embedding-3-small")
        return response.data[0].embedding

    def embed_many(self, texts: list):
        response = self.client.embeddings.create(input=texts, model="text-embedding-3-small")
        return [item.embedding for item in response.data]

    def chat(self, messages: list):
        response = self.client.chat.completions.create(
            model="gpt-6-luna",
            messages=messages
        )

        return response.choices[0].message.content