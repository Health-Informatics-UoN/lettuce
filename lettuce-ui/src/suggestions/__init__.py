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

    def accept_suggestion(self, index: int) -> AcceptedSuggestion:
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

        return AcceptedSuggestion(
                search_term=self.search_term,
                domains=self.domains,
                vocabs=self.vocabs,
                search_standard_concept=self.standard_concept,
                valid_concept=self.valid_concept,
                search_mode=self.search_mode,
                concept_id=self.suggestion[index].concept_id,
                concept_name=self.suggestion[index].concept_name,
                domain_id=self.suggestion[index].domain_id,
                vocabulary_id=self.suggestion[index].vocabulary_id,
                standard_concept=self.suggestion[index].standard_concept,
                score=self.suggestion[index].score,
                )

