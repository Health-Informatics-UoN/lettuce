from dataclasses import dataclass
from typing import List, Literal

@dataclass
class ConceptSuggestion:
    concept_id: int
    concept_name: str
    domain_id: str
    vocabulary_id: str
    standard_concept: str
    score: float = 0.0

@dataclass
class SuggestionRecord:
    search_term: str
    domains: List[str]
    vocabs: List[str]
    standard_concept: bool
    valid_concept: bool
    search_mode: Literal["text-search", "vector-search", "ai-search"]
    suggestion: ConceptSuggestion

@dataclass
class AcceptedSuggestion:
    search_term: str
    domains: List[str]
    vocabs: List[str]
    search_standard_concept: bool
    valid_concept: bool
    search_mode: Literal["text-search", "vector-search", "ai-search"]
    concept_id: int
    concept_name: str
    domain_id: str
    vocabulary_id: str
    standard_concept: str
    score: float = 0.0
