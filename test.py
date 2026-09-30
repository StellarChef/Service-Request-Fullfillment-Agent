from AI.openaiclient import AIClient
from AI.promptguard import PromptGuard, InjectionCheck, PromptInjectionError
from AI.openaiservice import AIService

client = AIClient()
guard = PromptGuard(AIClient())
service = AIService()


question = guard.check(client.chat([
    {"role": "system", "content": "You are a helpful assistant give me Complain Policy."},]))
print(question)

if __name__ == "__main__":
    pass