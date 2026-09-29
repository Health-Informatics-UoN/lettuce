from datetime import date
from typing import Self, Literal

from pydantic import BaseModel, Field

class ConceptDescription(BaseModel):
    concept_id: int
    concept_name: str
    domain_id: str
    concept_class_id: str
    vocabulary_id: str
    standard_concept: Literal["S", "C"] | None
    concept_code: str
    valid_start_date: date
    valid_end_date: date
    invalid_reason: str|None

class ConceptSynonym(BaseModel):
    """Model for concept synonym information"""
    concept_synonym_name: str
    concept_synonym_name_similarity_score: float


class ConceptRelationship(BaseModel):
    """Model for concept relationship information"""
    concept_id_1: int
    relationship_id: str
    concept_id_2: int

class RelatedConcept(BaseModel):
    """Model for related concept with relationship details"""
    concept: ConceptDescription
    relationship: ConceptRelationship

class AncestorRelationship(BaseModel):
    """Model for ancestor/descendant relationship information"""
    relationship_type: str
    ancestor_concept_id: int
    descendant_concept_id: int
    min_levels_of_separation: int
    max_levels_of_separation: int

class AncestorConcept(BaseModel):
    """Model for ancestor/descendant concept with relationship details"""
    concept: ConceptDescription
    relationship: AncestorRelationship

    @classmethod
    def from_row_mapping(cls, row_mapping) -> Self:
        return type(cls)(
                concept = ConceptDescription(
                    concept_name=row_mapping.concept_name,
                    concept_id=row_mapping.concept_id,
                    vocabulary_id=row_mapping.vocabulary_id,
                    concept_code=row_mapping.concept_code,
                ),
                relationship = AncestorRelationship(
                    relationship_type=row_mapping.relationship_type,
                    ancestor_concept_id=row_mapping.ancestor_concept_id,
                    descendant_concept_id=row_mapping.descendant_concept_id,
                    min_levels_of_separation=row_mapping.min_levels_of_separation,
                    max_levels_of_separation=row_mapping.max_levels_of_separation
                )
        )
        


class ConceptResult(BaseModel):
    """Model for OMOP concept search result"""
    concept: ConceptDescription
    concept_name_similarity_score: float
    concept_synonym: list[ConceptSynonym] = Field(default_factory=list)
    concept_ancestor: list[AncestorConcept] = Field(default_factory=list)
    concept_relationship: list[RelatedConcept] = Field(default_factory=list)
    

class SearchResult(BaseModel):
    """Model for search term result"""
    search_term: str
    concept: list[ConceptDescription] | None
