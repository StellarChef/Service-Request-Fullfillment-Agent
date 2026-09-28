from openai import OpenAI

client = OpenAI()


class OpenAICommunicator:
    def __init__(self):
        self.client = OpenAI()

    def embed_text(self, text: str):
        response = self.client.embeddings.create(input=f"{text}", model="text-embedding-3-small")
        return response.data[0].embedding

    def chat(self, messages: list):
        response = self.client.create(
            model="gpt-6-luna",
            input=messages
        )

        return response.output_text