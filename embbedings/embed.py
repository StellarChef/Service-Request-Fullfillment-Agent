from openai import OpenAI

client = OpenAI()


def embed_text(text: str):
    response = client.embeddings.create(input=f"{text}", model="text-embedding-3-small")
    return response.data[0].embedding
