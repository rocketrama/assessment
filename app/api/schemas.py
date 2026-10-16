"""The POST /chat contract (TASK.pdf §9.1). Field names and values must match exactly."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Route = Literal["rag", "tool", "hybrid", "clarify", "no_answer", "refuse"]
ToolStatus = Literal["pending_confirmation", "success", "error"]


class ChatRequest(BaseModel):
    # strict: no coercion of e.g. 123 → "123"; an invalid body is a 400
    model_config = ConfigDict(strict=True)

    session_id: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=4000)


class Citation(BaseModel):
    doc_id: str
    locator: str


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any]
    status: ToolStatus


class ChatResponse(BaseModel):
    request_id: str
    answer: str
    route: Route
    citations: list[Citation] = []
    tool_calls: list[ToolCall] = []
    pending_confirmation: bool = False
