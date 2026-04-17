from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class AgentDecision(str, Enum):
    ANSWER = "answer"
    SEARCH = "search_docs"
    CREATE_TICKET = "create_ticket"


class ChatRequest(BaseModel):
    conversation_id: str = Field(..., description="Conversation identifier")
    user_id: Optional[str] = None
    query: str = Field(..., min_length=1)
    prompt_version: str = "v1"


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    decision: AgentDecision
    confidence: float
    prompt_version: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    ticket_id: Optional[str] = None
    trace_id: Optional[str] = None


class DocumentChunk(BaseModel):
    chunk_id: str
    source: str
    section: str
    content: str
    keywords: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    chunk: DocumentChunk
    score: float


class ToolResult(BaseModel):
    name: str
    success: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class AgentState(BaseModel):
    conversation_id: str
    user_query: str
    prompt_version: str = "v1"
    history: list[dict[str, str]] = Field(default_factory=list)
    observations: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    max_steps: int = 4
    confidence_threshold: float = 0.62
    final_answer: Optional[str] = None
    decision: Optional[AgentDecision] = None
    ticket_id: Optional[str] = None


@dataclass
class PromptArtifact:
    version: str
    system: str
    user_template: str
    metadata: dict[str, Any] = field(default_factory=dict)
