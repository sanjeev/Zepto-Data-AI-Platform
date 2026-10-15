from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Customer question")


class AskResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list, description="Chunk ids used; empty for general questions")
    confidence: float = Field(..., ge=0.0, le=1.0)
