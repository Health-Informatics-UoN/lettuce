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
    
@dataclass
class SuggestionRecord:
    search_term: str
    domains: List[str]
    vocabs: List[str]
    standard_concept: bool
    valid_concept: bool
    search_mode: Literal["text-search", "vector-search", "ai-search"]
    suggestion: List[ConceptSuggestion]

def accept_suggestion(suggestions: SuggestionRecord, accepted: ConceptSuggestion) -> AcceptedSuggestion:
    """
    Select one of the suggestions and make an AcceptedSuggestion out of it 

    Parameters
    ----------
    index: int
        The index of the accepted suggestion

    Returns
    -------
    AcceptedSuggestion
        An AcceptedSuggestion with the details of this record and the chosen suggestion
    """
    if accepted in suggestions.suggestion:
        return AcceptedSuggestion(
                search_term=suggestions.search_term,
                domains=suggestions.domains,
                vocabs=suggestions.vocabs,
                search_standard_concept=suggestions.standard_concept,
                valid_concept=suggestions.valid_concept,
                search_mode=suggestions.search_mode,
                concept_id=accepted.concept_id,
                concept_name=accepted.concept_name,
                domain_id=accepted.domain_id,
                vocabulary_id=accepted.vocabulary_id,
                standard_concept=accepted.standard_concept,
                score=accepted.score,
                )
    else:
        raise KeyError

