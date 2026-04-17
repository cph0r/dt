from __future__ import annotations

from fastapi import APIRouter, Depends

from app.models.schemas import ChatRequest, ChatResponse
from app.services.agent import Agent

router = APIRouter(prefix="/v1", tags=["support"])


def get_agent() -> Agent:
    raise RuntimeError("Agent dependency not configured")


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, agent: Agent = Depends(get_agent)) -> ChatResponse:
    return agent.run(
        conversation_id=request.conversation_id,
        query=request.query,
        prompt_version=request.prompt_version,
    )
