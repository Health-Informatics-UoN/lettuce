from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class Suggestion(BaseModel):
    concept_name: str
    concept_id: int
    domain_id: str
    vocabulary_id: str
    concept_class_id: str
    standard_concept: Optional[str]
    invalid_reason: Optional[str]
    ranks: Optional[Dict[str, int]]
    scores: Optional[Dict[str, float]]

class SuggestionsMetaData(BaseModel):
    assistant: str = "Lettuce"
    version: str = "0.1.0"
    pipeline: Optional[str] = None

class ConceptSuggestionResponse(BaseModel):
    recommendations: List[Suggestion]
    metadata: SuggestionsMetaData = Field(default_factory=SuggestionsMetaData)
