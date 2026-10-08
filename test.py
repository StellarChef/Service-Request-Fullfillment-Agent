from AI.openaiclient import AIClient
from AI.promptguard import PromptGuard, PromptInjectionError
from AI.openaiservice import AIService
from AI.clasification import AIClassification

client = AIClient()
guard = PromptGuard(client)
clasify = AIClassification(client)
service = AIService(client, guard, clasify)


if __name__ == "__main__":
    text = "Zwracam się z uprzejmą prośbą, chciałabym dowiedzieć się jak mogę poprawnie dokonać wymiarów mojej sylwetki aby zamówić personalizowany Trencz"

    try:
        service.verify_injection(text)
        print(service.classify_request(text))
    except PromptInjectionError as e:
        print("Blocked:", e)
