from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from app.evaluation.evaluator import Evaluator
from app.models.schemas import AgentDecision, AgentState, ChatResponse
from app.prompts.registry import get_prompt
from app.services.llm import LLMClient
from app.services.memory import ConversationMemory
from app.tools.base import ToolRegistry
from app.utils.logging import get_logger


@dataclass
class Agent:
    llm: LLMClient
    tools: ToolRegistry
    evaluator: Evaluator
    memory: ConversationMemory = field(default_factory=ConversationMemory)
    max_steps: int = 4
    confidence_threshold: float = 0.62
    retrieval_top_k: int = 4
    enable_reranking: bool = False
    planner_model: str | None = None
    answer_model: str | None = None
    logger_name: str = "support.agent"

    def run(self, conversation_id: str, query: str, prompt_version: str = "v1") -> ChatResponse:
        logger = get_logger(self.logger_name)
        start = time.monotonic()
        prompt = get_prompt(prompt_version)
        snapshot = self.memory.get_snapshot(conversation_id)
        state = AgentState(
            conversation_id=conversation_id,
            user_query=query,
            prompt_version=prompt_version,
            history=snapshot.turns,
            max_steps=self.max_steps,
            confidence_threshold=self.confidence_threshold,
        )
        evidence: list[dict[str, Any]] = []
        answer = ""
        decision = AgentDecision.CREATE_TICKET
        tool_calls: list[dict[str, Any]] = []
        ticket_id: str | None = None
        confidence = 0.0
        adjusted_confidence = 0.0

        for step in range(1, state.max_steps + 1):
            logger.info(
                "agent_planning",
                extra={
                    "event": "agent_step",
                    "phase": "planning",
                    "step": step,
                    "conversation_id": conversation_id,
                    "prompt_version": prompt_version,
                },
            )
            llm_start = time.monotonic()
            try:
                llm_payload = self.llm.complete_with_model(
                    self.planner_model or self.llm.model,
                    [
                        {"role": "system", "content": prompt.system},
                        {
                            "role": "user",
                            "content": prompt.user_template.format(
                                context=self._build_memory_context(snapshot.summary, snapshot.turns),
                                query=query,
                            ),
                        },
                    ]
                )
            except Exception:
                logger.warning(
                    "llm_failed_falling_back_to_ticket",
                    extra={
                        "event": "failure",
                        "phase": "planning",
                        "step": step,
                        "conversation_id": conversation_id,
                        "prompt_version": prompt_version,
                        "failure_reason": "llm_call_failed",
                    },
                )
                ticket_tool = self.tools.get("create_ticket")
                tool_start = time.monotonic()
                ticket_result = ticket_tool.execute(issue=query)
                tool_latency_ms = round((time.monotonic() - tool_start) * 1000, 2)
                ticket_id = ticket_result.data.get("ticket_id")
                tool_calls.append({"name": ticket_tool.name, "arguments": {"issue": query}, "result": ticket_result.data})
                answer = "I could not confidently resolve this from the knowledge base, so I created a support ticket."
                decision = AgentDecision.CREATE_TICKET
                logger.info(
                    "agent_tool_execution",
                    extra={
                        "event": "agent_step",
                        "phase": "execution",
                        "step": step,
                        "conversation_id": conversation_id,
                        "prompt_version": prompt_version,
                        "selected_tool": ticket_tool.name,
                        "tool_name": ticket_tool.name,
                        "latency_ms": tool_latency_ms,
                    },
                )
                break

            plan_latency_ms = round((time.monotonic() - llm_start) * 1000, 2)

            decision = self._coerce_decision(llm_payload.get("decision"))
            confidence = float(llm_payload.get("confidence", 0.0))
            logger.info(
                "agent_step",
                extra={
                    "event": "agent_step",
                    "phase": "planning",
                    "step": step,
                    "conversation_id": conversation_id,
                    "prompt_version": prompt_version,
                    "decision": decision.value,
                    "confidence": confidence,
                    "query": query,
                },
            )
            logger.info(
                "agent_planning_result",
                extra={
                    "event": "agent_step",
                    "phase": "planning",
                    "step": step,
                    "conversation_id": conversation_id,
                    "prompt_version": prompt_version,
                    "decision": decision.value,
                    "confidence": confidence,
                    "latency_ms": plan_latency_ms,
                },
            )

            if decision == AgentDecision.SEARCH:
                logger.info(
                    "agent_tool_selected",
                    extra={
                        "event": "agent_step",
                        "phase": "tool_selection",
                        "step": step,
                        "conversation_id": conversation_id,
                        "prompt_version": prompt_version,
                        "selected_tool": "search_docs",
                    },
                )
                tool = self.tools.get("search_docs")
                tool_start = time.monotonic()
                tool_result = tool.execute(query=query, top_k=self.retrieval_top_k, rerank=self.enable_reranking)
                tool_latency_ms = round((time.monotonic() - tool_start) * 1000, 2)
                evidence = tool_result.data.get("results", [])
                tool_calls.append(
                    {
                        "name": tool.name,
                        "arguments": {"query": query, "top_k": self.retrieval_top_k, "rerank": self.enable_reranking},
                        "result": tool_result.data,
                    }
                )
                logger.info(
                    "agent_tool_execution",
                    extra={
                        "event": "agent_step",
                        "phase": "execution",
                        "step": step,
                        "conversation_id": conversation_id,
                        "prompt_version": prompt_version,
                        "selected_tool": tool.name,
                        "tool_name": tool.name,
                        "latency_ms": tool_latency_ms,
                    },
                )
                logger.info(
                    "agent_observation",
                    extra={
                        "event": "agent_step",
                        "phase": "observation",
                        "step": step,
                        "conversation_id": conversation_id,
                        "prompt_version": prompt_version,
                        "decision": "search_docs",
                        "confidence": confidence,
                    },
                )
                if evidence:
                    candidate_answer = self._synthesize_answer(query, evidence)
                    judged = self.evaluator.judge(query, candidate_answer, evidence)
                    adjusted_confidence = self.evaluator.adjusted_confidence(confidence, judged)
                    should_fallback = self.evaluator.should_fallback(adjusted_confidence, state.confidence_threshold)
                    logger.info(
                        "agent_evaluation",
                        extra={
                            "event": "agent_step",
                            "phase": "observation",
                            "step": step,
                            "conversation_id": conversation_id,
                            "prompt_version": prompt_version,
                            "decision": "answer_candidate",
                            "confidence": confidence,
                            "adjusted_confidence": adjusted_confidence,
                            "evaluator_score": judged.score,
                        },
                    )
                    if not should_fallback and judged.passed:
                        answer = candidate_answer
                        decision = AgentDecision.ANSWER
                        break
                    answer = ""
                    decision = AgentDecision.CREATE_TICKET
                continue

            if decision == AgentDecision.ANSWER:
                answer = str(llm_payload.get("answer", ""))
                judged = self.evaluator.judge(query, answer, evidence)
                adjusted_confidence = self.evaluator.adjusted_confidence(confidence, judged)
                logger.info(
                    "agent_observation",
                    extra={
                        "event": "agent_step",
                        "phase": "observation",
                        "step": step,
                        "conversation_id": conversation_id,
                        "prompt_version": prompt_version,
                        "decision": "answer_candidate",
                        "confidence": confidence,
                        "adjusted_confidence": adjusted_confidence,
                        "evaluator_score": judged.score,
                    },
                )
                if not self.evaluator.should_fallback(adjusted_confidence, state.confidence_threshold) and judged.passed:
                    confidence = adjusted_confidence
                    break
                answer = ""
                decision = AgentDecision.CREATE_TICKET
                continue

            logger.info(
                "agent_tool_selected",
                extra={
                    "event": "agent_step",
                    "phase": "tool_selection",
                    "step": step,
                    "conversation_id": conversation_id,
                    "prompt_version": prompt_version,
                    "selected_tool": "create_ticket",
                },
            )
            ticket_tool = self.tools.get("create_ticket")
            tool_start = time.monotonic()
            ticket_result = ticket_tool.execute(issue=query)
            tool_latency_ms = round((time.monotonic() - tool_start) * 1000, 2)
            ticket_id = ticket_result.data.get("ticket_id")
            tool_calls.append({"name": ticket_tool.name, "arguments": {"issue": query}, "result": ticket_result.data})
            answer = "I could not confidently resolve this from the knowledge base, so I created a support ticket."
            decision = AgentDecision.CREATE_TICKET
            logger.info(
                "agent_tool_execution",
                extra={
                    "event": "agent_step",
                    "phase": "execution",
                    "step": step,
                    "conversation_id": conversation_id,
                    "prompt_version": prompt_version,
                    "selected_tool": ticket_tool.name,
                    "tool_name": ticket_tool.name,
                    "latency_ms": tool_latency_ms,
                },
            )
            break

        if not answer:
            ticket_tool = self.tools.get("create_ticket")
            tool_start = time.monotonic()
            ticket_result = ticket_tool.execute(issue=query)
            tool_latency_ms = round((time.monotonic() - tool_start) * 1000, 2)
            ticket_id = ticket_result.data.get("ticket_id")
            tool_calls.append({"name": ticket_tool.name, "arguments": {"issue": query}, "result": ticket_result.data})
            answer = "I could not confidently resolve this from the knowledge base, so I created a support ticket."
            decision = AgentDecision.CREATE_TICKET
            logger.info(
                "agent_tool_execution",
                extra={
                    "event": "agent_step",
                    "phase": "execution",
                    "step": state.max_steps,
                    "conversation_id": conversation_id,
                    "prompt_version": prompt_version,
                    "selected_tool": ticket_tool.name,
                    "tool_name": ticket_tool.name,
                    "latency_ms": tool_latency_ms,
                },
            )

        self.memory.append(conversation_id, "user", query)
        self.memory.append(conversation_id, "assistant", answer)

        latency_ms = round((time.monotonic() - start) * 1000, 2)
        logger.info(
            "agent_completed",
            extra={
                "event": "agent_completed",
                "conversation_id": conversation_id,
                "prompt_version": prompt_version,
                "latency_ms": latency_ms,
                "decision": decision.value,
                "confidence": confidence,
                "adjusted_confidence": adjusted_confidence,
            },
        )
        return ChatResponse(
            conversation_id=conversation_id,
            answer=answer,
            decision=decision,
            confidence=adjusted_confidence or confidence,
            prompt_version=prompt_version,
            tool_calls=tool_calls,
            ticket_id=ticket_id,
        )

    def _compose_answer(self, query: str, evidence: list[dict[str, Any]]) -> str:
        return self._synthesize_answer(query, evidence)

    def _synthesize_answer(self, query: str, evidence: list[dict[str, Any]]) -> str:
        context_lines: list[str] = []
        for idx, item in enumerate(evidence[:5], start=1):
            context_lines.append(
                f"[{idx}] source={item.get('source')} section={item.get('section')} content={item.get('content', '')[:500]}"
            )
        synthesis_prompt = (
            "SYNTHESIZE_ANSWER\n"
            "You are generating a grounded support answer from retrieved evidence. "
            "Cite the most relevant source and section names in plain text.\n\n"
            f"Question: {query}\n"
            f"Evidence:\n{chr(10).join(context_lines)}"
        )
        payload = self.llm.complete_with_model(
            self.answer_model or self.llm.model,
            [
                {"role": "system", "content": "Return concise, factual support answers grounded in evidence."},
                {"role": "user", "content": synthesis_prompt},
            ],
        )
        return str(payload.get("answer") or payload.get("raw") or "I found relevant documentation but could not synthesize a full answer.")

    def _build_memory_context(self, summary: str, history: list[dict[str, str]]) -> str:
        lines: list[str] = []
        if summary:
            lines.append(f"conversation_summary: {summary}")
        lines.extend(f"{item['role']}: {item['content']}" for item in history)
        return "\n".join(lines)

    def _coerce_decision(self, value: Any) -> AgentDecision:
        try:
            return AgentDecision(str(value))
        except Exception:
            return AgentDecision.CREATE_TICKET
