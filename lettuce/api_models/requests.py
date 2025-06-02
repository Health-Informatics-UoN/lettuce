from typing import Optional, List
from pydantic import BaseModel

class ConceptSuggestionRequest(BaseModel):
    """
    A model describing API requests for concept suggestions

    Attributes
    ----------
    source_term: str
        The source term the request wants concept suggestions for
    vocabulary_id: Optional[List[str]]
        An optional filter on the vocabularies searched.
        If None, no filter is applied.
        If any vocabulary_id are supplied, only concepts from those
        vocabularies will be suggested
    domain_id: Optional[List[str]]
        An optional filter on the domains searched.
        If None, no filter is applied.
        If any vocabulary_id are supplied, only concepts from those
        domains will be suggested
    standard_concept: bool
        Filter on standard concepts.
        If True, only standard concepts will be suggested
    valid_concept: bool
        Filter on valid concepts.
        If True, only valid concepts will be suggested
    top_k: int
        The number of suggestions to make
    """
    source_term: str
    vocabulary_id: Optional[List[str]] = None
    domain_id: Optional[List[str]] = None
    standard_concept: bool = False
    valid_concept: bool = False
    top_k: int = 5
