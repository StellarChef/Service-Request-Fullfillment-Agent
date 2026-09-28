from AI.openaiclient import AIClient
from AI.chunker import Chunker


openai_client = AIClient()
chunker = Chunker()

class AIService:
    def __init__(self, client=openai_client):
        self.client = client

    def _complaint_input(self, complaint_text):
        complaint_vector = self.client.embed_text(complaint_text)
        return complaint_vector

    