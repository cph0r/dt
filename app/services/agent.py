from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from app.evaluation.evaluator import Evaluator
from app.models.schemas import AgentDecision, AgentState, ChatResponse
from app.prompts.registry import get_prompt
from app.services.llm import LLMClient
from app.tools.base import ToolRegistry
from app.utils.logging import get_logger


@dataclass
class Agent:
    llm: LLMClient
    tools: ToolRegistry
    evaluator: Evaluator
    max_steps: int = 4
    confidence_threshold: float = 0.62
    logger_name: str = "support.agent"
    conversation_store: dict[str, list[dict[str, str]]] = field(default_factory=dict)

    def run(self, conversation_id: str, query: str, prompt_version: str = "v1") -> ChatResponse:
        logger = get_logger(self.logger_name)
        start = time.monotonic()
        prompt = get_prompt(prompt_version)
        history = self.conversation_store.get(conversation_id, [])
        state = AgentState(
            conversation_id=conversation_id,
            user_query=query,
            prompt_version=prompt_version,
            history=history,
            max_steps=self.max_steps,
            confidence_threshold=self.confidence_threshold,
        )
        evidence: list[dict[str, Any]] = []
        answer = ""
        decision = AgentDecision.CREATE_TICKET
        tool_calls: list[dict[str, Any]] = []
        ticket_id: str | None = None
        confidence = 0.0

        for _ in range(state.max_steps):
            try:
                llm_payload = self.llm.complete(
                    [
                        {"role": "system", "content": prompt.system},
                        {"role": "user", "content": prompt.user_template.format(context=self._format_history(history), query=query)},
                    ]
                )
            except Exception as exc:
                logger.warning(
                    "llm_failed_falling_back_to_ticket",
                    extra={"event": "llm_failed", "conversation_id": conversation_id, "prompt_version": prompt_version},
                )
                ticket_tool = self.tools.get("create_ticket")
                ticket_result = ticket_tool.execute(issue=query)
                ticket_id = ticket_result.data.get("ticket_id")
                tool_calls.append({"name": ticket_tool.name, "arguments": {"issue": query}, "result": ticket_result.data})
                answer = "I could not confidently resolve this from the knowledge base, so I created a support ticket."
                decision = AgentDecision.CREATE_TICKET
                break

            decision = self._coerce_decision(llm_payload.get("decision"))
            confidence = float(llm_payload.get("confidence", 0.0))

            if decision == AgentDecision.SEARCH:
                tool = self.tools.get("search_docs")
                tool_result = tool.execute(query=query, top_k=4)
                evidence = tool_result.data.get("results", [])
                tool_calls.append({"name": tool.name, "arguments": {"query": query}, "result": tool_result.data})
                if evidence and confidence >= state.confidence_threshold:
                    answer = self._compose_answer(query, evidence)
                    decision = AgentDecision.ANSWER
                    break
                continue

            if decision == AgentDecision.ANSWER:
                answer = str(llm_payload.get("answer", ""))
                judged = self.evaluator.judge(query, answer, evidence)
                if judged.passed or confidence >= state.confidence_threshold:
                    break
                decision = AgentDecision.CREATE_TICKET
                continue

            ticket_tool = self.tools.get("create_ticket")
            ticket_result = ticket_tool.execute(issue=query)
            ticket_id = ticket_result.data.get("ticket_id")
            tool_calls.append({"name": ticket_tool.name, "arguments": {"issue": query}, "result": ticket_result.data})
            answer = "I could not confidently resolve this from the knowledge base, so I created a support ticket."
            decision = AgentDecision.CREATE_TICKET
            break

        if not answer:
            ticket_tool = self.tools.get("create_ticket")
            ticket_result = ticket_tool.execute(issue=query)
            ticket_id = ticket_result.data.get("ticket_id")
            tool_calls.append({"name": ticket_tool.name, "arguments": {"issue": query}, "result": ticket_result.data})
            answer = "I could not confidently resolve this from the knowledge base, so I created a support ticket."
            decision = AgentDecision.CREATE_TICKET

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})
        self.conversation_store[conversation_id] = history

        latency_ms = round((time.monotonic() - start) * 1000, 2)
        logger.info(
            "agent_completed",
            extra={
                "event": "agent_completed",
                "conversation_id": conversation_id,
                "prompt_version": prompt_version,
                "latency_ms": latency_ms,
            },
        )
        return ChatResponse(
            conversation_id=conversation_id,
            answer=answer,
            decision=decision,
            confidence=float(llm_payload.get("confidence", 0.0)) if 'llm_payload' in locals() else 0.0,
            prompt_version=prompt_version,
            tool_calls=tool_calls,
            ticket_id=ticket_id,
        )

    def _compose_answer(self, query: str, evidence: list[dict[str, Any]]) -> str:
        top = evidence[0]
        source = top.get("source", "knowledge base")
        section = top.get("section", "general")
        content = top.get("content", "")
        return f"Based on {source} / {section}: {content[:250]}"

    def _format_history(self, history: list[dict[str, str]]) -> str:
        return "\n".join(f"{item['role']}: {item['content']}" for item in history[-6:])

    def _coerce_decision(self, value: Any) -> AgentDecision:
        try:
            return AgentDecision(str(value))
        except Exception:
            return AgentDecision.CREATE_TICKET
