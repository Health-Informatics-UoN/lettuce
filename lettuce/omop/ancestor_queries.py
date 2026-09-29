from pydantic import BaseModel
from sqlalchemy import Select, select

from omop.omop_models import Concept, ConceptAncestor
from omop.result_models import ConceptDescription

class HierarchyEdge(BaseModel):
    descendant_concept_id: int
    ancestor_concept_id:int

class HierarchyGraph(BaseModel):
    concepts: list[ConceptDescription]
    adjacency_list: list[HierarchyEdge]

def hierarchy_adjacency_query(concept_set_query: Select) -> Select:
    """
    Build a query to assemble an adjacency list for the hierarchy for a query returning concept_ids

    Parameters
    ----------
    concept_set_query: Select
        A SQLalchemy select for a query defining a set of concepts

    Returns
    -------
    Select
        SQLalchemy Select object for ancestor and descendant concepts
    """
    return (
        select(
        ConceptAncestor.ancestor_concept_id, ConceptAncestor.descendant_concept_id
        )
        .where(ConceptAncestor.descendant_concept_id.in_(concept_set_query))
        .where(ConceptAncestor.min_levels_of_separation == 1)
    )

def ancestor_adjacency_query(concept_id: int) -> Select:
    """
    Build a query to assemble an adjacency list for the hierarchy of ancestors to a given concept_id

    Parameters
    ----------
    concept_id: int
        The concept_id for a concept for which you want the ancestors

    Returns
    -------
    Select
        SQLalchemy Select object for ancestor and descendant concepts
    """
    return hierarchy_adjacency_query(
            select(ConceptAncestor.ancestor_concept_id).where(
        ConceptAncestor.descendant_concept_id == concept_id
    ))

def descendant_adjacency_query(concept_id: int) -> Select:
    """
    Build a query to assemble an adjacency list for the hierarchy of descendants of a given concept_id

    Parameters
    ----------
    concept_id: int
        The concept_id for a concept for which you want the ancestors

    Returns
    -------
    Select
        SQLalchemy Select object for ancestor and descendant concepts
    """
    return hierarchy_adjacency_query(select(ConceptAncestor.descendant_concept_id).where(
            ConceptAncestor.ancestor_concept_id == concept_id
            ))

def bulk_concept_by_id_query(concept_ids: list[int]) -> Select:
    """
    Build a query to fetch lots of concepts by ID.

    Parameters
    ----------
    concept_ids: list[int]
        A list of concept_ids to fetch

    Returns
    -------
    Select
        A query for retrieving the concept_ids
    """
    return select(Concept).where(
            Concept.concept_id.in_(concept_ids)
            )
