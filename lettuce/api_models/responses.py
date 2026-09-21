from typing import Any, Optional

from pydantic import BaseModel, Field


class Suggestion(BaseModel):
    conceptName: str
    conceptId: int
    conceptCode: str
    domain: str
    vocabulary: str
    conceptClass: str
    standard_concept: Optional[str]
    invalid_reason: Optional[str]
    ranks: Optional[dict[str, int]]
    scores: Optional[dict[str, float]]


class SuggestionsMetaData(BaseModel):
    assistant: str = "Lettuce"
    version: str = "0.1.0"
    pipeline: Optional[str] = None
    info: Optional[dict[str, Any]] = None


class ConceptSuggestionResponse(BaseModel):
    items: list[Suggestion]
    metadata: SuggestionsMetaData = Field(default_factory=SuggestionsMetaData)
