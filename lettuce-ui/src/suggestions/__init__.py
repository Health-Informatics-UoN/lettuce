from dataclasses import dataclass
from typing import List, Literal

@dataclass
class ConceptSuggestion:
    """
    A dataclass to hold a suggested concept

    Attributes
    ----------
    concept_id: int
        The concept_id from the concept table
    concept_name: str
        The concept_name from the concept table
    domain_id: str
        The domain_id from the concept table
    vocabulary_id: str
        The vocabulary_id from the concept table
    standard_concept: str
        The standard_concept from the concept table
    score: float
        If the concept has been retrieved with a score, this is it, else 0.0
    """
    concept_id: int
    concept_name: str
    domain_id: str
    vocabulary_id: str
    standard_concept: str
    score: float = 0.0

@dataclass
class AcceptedSuggestion:
    """
    A dataclass to hold a suggestion accepted by a user

    Attributes
    ----------
    search_term: str
        The search term used to find the accepted concept
    domains: List[str]
        The domains used to find the accepted concept
    vocabs: List[str]
        The vocabularies used to find the accepted concept
    search_standard_concept: bool
        Whether only standard concepts were included in search
    valid_concept: bool
        Whether only valid concepts were included in search
    search_mode: Literal["text-search", "vector-search", "ai-search"]
        The search mode used to find the concept
    concept_id: int
        The concept_id from the concept table
    concept_name: str
        The concept_name from the concept table
    domain_id: str
        The domain_id from the concept table
    vocabulary_id: str
        The vocabulary_id from the concept table
    standard_concept: str
        The standard_concept from the concept table
    score: float
        If the concept has been retrieved with a score, this is it, else 0.0
    """
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
    """
    A dataclass to hold the concepts retrieved by a search

    Attributes
    ----------
    search_term: str
        The search term used in search
    domains: List[str]
        The domains included in search
    vocabs: List[str]
        The vocabularies included in search
    standard_concept: bool
        Whether only standard concepts were included in search 
    valid_concept: bool
        Whether only valid concepts were included in search
    search_mode: Literal["text-search", "vector-search", "ai-search"]
        The search mode used to find concepts
    suggestion: List[ConceptSuggestion]
        Details of the concepts in search results
    """
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

