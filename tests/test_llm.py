from app.services.llm import LLMClient, LiteLLMBackend


class JsonBackend(LiteLLMBackend):
    def complete(self, model, messages, temperature=0.0):
        return "```json\n{\"decision\": \"answer\", \"confidence\": 0.91, \"answer\": \"ok\"}\n```"


def test_llm_client_extracts_json_from_fenced_block():
    client = LLMClient(JsonBackend(provider="mock"), model="mock://support", retry_count=0, timeout_s=1)

    payload = client.complete([{"role": "user", "content": "hello"}])

    assert payload["decision"] == "answer"
    assert payload["confidence"] == 0.91
