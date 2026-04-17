from pathlib import Path
from tempfile import TemporaryDirectory

from app.evaluation.evaluator import Evaluator
from app.models.schemas import AgentDecision
from app.services.agent import Agent
from app.services.llm import LLMClient, LiteLLMBackend
from app.services.retriever import Retriever
from app.services.vector_store import SQLiteVectorStore
from app.tools.base import ToolRegistry
from app.tools.knowledge_base import SearchDocsTool
from app.tools.ticketing import CreateTicketTool
from app.ingestion.chunking import ChunkingPipeline


class LowConfidenceBackend(LiteLLMBackend):
    def complete(self, model, messages, temperature=0.0):
        return '{"decision":"search_docs","confidence":0.4,"answer":"","tool":null}'


class OverconfidentAnswerBackend(LiteLLMBackend):
    def complete(self, model, messages, temperature=0.0):
        return '{"decision":"answer","confidence":0.99,"answer":"ok","tool":null}'


def build_agent():
    temp_dir = TemporaryDirectory()
    database_path = Path(temp_dir.name) / "agent.sqlite3"
    retriever = Retriever(store=SQLiteVectorStore(str(database_path)))
    retriever.index(ChunkingPipeline().chunk("kb", "# Password Reset\n\nUse the reset link on the sign in page."))
    tools = ToolRegistry()
    tools.register(SearchDocsTool(retriever=retriever))
    tools.register(CreateTicketTool())
    agent = Agent(
        llm=LLMClient(LowConfidenceBackend(), model="test", retry_count=0, timeout_s=1),
        tools=tools,
        evaluator=Evaluator(threshold=0.6),
        max_steps=2,
        confidence_threshold=0.8,
    )
    agent._temp_dir = temp_dir
    return agent


def test_agent_falls_back_to_ticket_when_confidence_low():
    agent = build_agent()
    response = agent.run("conv-1", "I cannot log in and nothing works", "v1")

    assert response.decision == AgentDecision.CREATE_TICKET
    assert response.ticket_id is not None
    assert response.tool_calls


def test_agent_falls_back_when_evaluator_reduces_confidence():
    temp_dir = TemporaryDirectory()
    database_path = Path(temp_dir.name) / "agent-eval.sqlite3"
    retriever = Retriever(store=SQLiteVectorStore(str(database_path)))
    tools = ToolRegistry()
    tools.register(SearchDocsTool(retriever=retriever))
    tools.register(CreateTicketTool())
    agent = Agent(
        llm=LLMClient(OverconfidentAnswerBackend(), model="test", retry_count=0, timeout_s=1),
        tools=tools,
        evaluator=Evaluator(threshold=0.8),
        max_steps=2,
        confidence_threshold=0.85,
    )
    response = agent.run("conv-2", "A complex unknown issue", "v1")

    assert response.decision == AgentDecision.CREATE_TICKET
    assert response.confidence < 0.85
