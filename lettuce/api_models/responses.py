from typing import Any, Optional

from pydantic import BaseModel, Field


class Suggestion(BaseModel):
    conceptName: str
    conceptId: int
    conceptCode: str
    domain: str
    vocabulary: str
    conceptClass: str
    standard_concept: str | None
    invalid_reason: str | None
    ranks: dict[str, int] | None
    scores: dict[str, float] | None


class SuggestionsMetaData(BaseModel):
    assistant: str = "Lettuce"
    version: str = "0.1.0"
    pipeline: str | None = None
    info: dict[str, Any] | None = None


class ConceptSuggestionResponse(BaseModel):
    items: list[Suggestion]
    metadata: SuggestionsMetaData = Field(default_factory=SuggestionsMetaData)
