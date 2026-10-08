from AI.clasification import AIClassification
from AI.openaiclient import AIClient
from AI.promptguard import PromptGuard
from models.schemas import InputClassificationModel


# ✅ / ✖️

class AIService:
    def __init__(self, client: AIClient, guard: PromptGuard, classifier: AIClassification):
        self.client = client
        self.guard = guard
        self.classifier = classifier

    def handle_request(self, text: str) -> str:
        """Runs the whole pipeline: guard -> classification -> policy matching -> reply."""
        pass
#✅
    def verify_injection(self, text: str) -> None:
        """Step 1: first-line injection check. Raises PromptInjectionError if the text is rejected."""
        return self.guard.validate(text)

#✅
    def classify_request(self, text: str) -> InputClassificationModel:
        """Step 2: assigns the request to COMPLAINT, INQUIRY or SPAM."""
        return self.classifier.classify(text)

    def match_policies(self, text: str, classification: InputClassificationModel) -> list[str]:
        """Step 3: verifies the topic and returns the policy fragments that apply to the request."""
        pass

    def generate_reply(self, text: str, classification: InputClassificationModel, policies: list[str]) -> str:
        """Step 4: writes the reply to the customer based on the matched policies."""
        pass

    def _complaint_input(self, complaint_text):
        complaint_vector = self.client.embed_text(complaint_text)
        return complaint_vector
