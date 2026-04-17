from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router
from app.api.routes import get_agent
from app.config.settings import get_settings
from app.evaluation.evaluator import Evaluator
from app.ingestion.chunking import ChunkingPipeline
from app.models.schemas import DocumentChunk
from app.prompts.registry import get_prompt
from app.services.agent import Agent
from app.services.llm import LLMClient, LiteLLMBackend
from app.services.retriever import Retriever
from app.tools.base import ToolRegistry
from app.tools.knowledge_base import SearchDocsTool
from app.tools.ticketing import CreateTicketTool
from app.utils.logging import configure_logging


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    retriever = Retriever()
    chunker = ChunkingPipeline()
    sample_docs = chunker.chunk(
        source="help_center",
        text=(
            "# Refunds\n\nRefunds are processed within 5 business days.\n\n"
            "# Password Reset\n\nUse the reset link on the sign in page to reset your password."
        ),
    )
    retriever.index(sample_docs)

    tools = ToolRegistry()
    tools.register(SearchDocsTool(retriever=retriever))
    tools.register(CreateTicketTool())

    llm = LLMClient(
        backend=LiteLLMBackend(),
        model=settings.llm_model,
        retry_count=settings.llm_retry_count,
        timeout_s=settings.llm_timeout_s,
    )
    agent = Agent(
        llm=llm,
        tools=tools,
        evaluator=Evaluator(),
        max_steps=settings.max_agent_steps,
        confidence_threshold=settings.confidence_threshold,
    )

    app = FastAPI(title=settings.app_name)
    app.dependency_overrides[get_agent] = lambda: agent
    app.include_router(router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "prompt_version": get_prompt(settings.prompt_version).version}

    app.state.agent = agent
    return app


app = create_app()
