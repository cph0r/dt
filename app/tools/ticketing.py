from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from app.models.schemas import ToolResult
from app.tools.base import Tool


@dataclass
class CreateTicketTool(Tool):
    name: str = "create_ticket"

    def execute(self, **kwargs: Any) -> ToolResult:
        issue = str(kwargs.get("issue", ""))
        ticket_id = f"TKT-{uuid.uuid4().hex[:10].upper()}"
        return ToolResult(name=self.name, success=True, data={"ticket_id": ticket_id, "issue": issue})
