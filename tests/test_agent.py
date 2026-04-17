from app.evaluation.evaluator import Evaluator
from app.models.schemas import AgentDecision
from app.services.agent import Agent
from app.services.llm import LLMClient, LiteLLMBackend
from app.services.retriever import Retriever
from app.tools.base import ToolRegistry
from app.tools.knowledge_base import SearchDocsTool
from app.tools.ticketing import CreateTicketTool
from app.ingestion.chunking import ChunkingPipeline


class LowConfidenceBackend(LiteLLMBackend):
    def complete(self, model, messages, temperature=0.0):
        return '{"decision":"search_docs","confidence":0.4,"answer":"","tool":null}'


def build_agent():
    retriever = Retriever()
    retriever.index(ChunkingPipeline().chunk("kb", "# Password Reset\n\nUse the reset link on the sign in page."))
    tools = ToolRegistry()
    tools.register(SearchDocsTool(retriever=retriever))
    tools.register(CreateTicketTool())
    return Agent(
        llm=LLMClient(LowConfidenceBackend(), model="test", retry_count=0, timeout_s=1),
        tools=tools,
        evaluator=Evaluator(threshold=0.6),
        max_steps=2,
        confidence_threshold=0.8,
    )


def test_agent_falls_back_to_ticket_when_confidence_low():
    agent = build_agent()
    response = agent.run("conv-1", "I cannot log in and nothing works", "v1")

    assert response.decision == AgentDecision.CREATE_TICKET
    assert response.ticket_id is not None
    assert response.tool_calls
