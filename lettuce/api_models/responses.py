from typing import Any, Dict, List, Optional
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
    ranks: Optional[Dict[str, int]]
    scores: Optional[Dict[str, float]]

class SuggestionsMetaData(BaseModel):
    assistant: str = "Lettuce"
    version: str = "0.1.0"
    pipeline: Optional[str] = None
    info: Optional[Dict[str, Any]] = None

class ConceptSuggestionResponse(BaseModel):
    items: List[Suggestion]
    metadata: SuggestionsMetaData = Field(default_factory=SuggestionsMetaData)
