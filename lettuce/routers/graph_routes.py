from fastapi import APIRouter

from omop.ancestor_queries import (
    ConceptDescription,
    HierarchyEdge,
    HierarchyGraph,
    ancestor_adjacency_query,
    bulk_concept_by_id_query,
    descendant_adjacency_query,
)
from omop.db_manager import get_session
from options.base_options import BaseOptions

settings = BaseOptions()

router = APIRouter()


@router.get("/ancestors/{concept_id}")
async def concept_ancestors(
    concept_id: int,
) -> HierarchyGraph:
    ancestor_query = ancestor_adjacency_query(concept_id)
    with get_session() as session:
        ancestors = session.execute(ancestor_query).fetchall()
    adjacency_list = [
        HierarchyEdge(
            descendant_concept_id=rel.descendant_concept_id,
            ancestor_concept_id=rel.ancestor_concept_id,
        )
        for rel in ancestors
    ]
    concept_set = set()
    for edge in adjacency_list:
        concept_set.add(edge.ancestor_concept_id)
        concept_set.add(edge.descendant_concept_id)
    concept_query = bulk_concept_by_id_query(list(concept_set))
    with get_session() as session:
        concepts = [
                ConceptDescription.model_validate(res._mapping)
                for res in 
                session.execute(concept_query).all()
                ]
    return HierarchyGraph(concepts=concepts, adjacency_list=adjacency_list)

@router.get("/descendants/{concept_id}")
async def concept_descendants(
    concept_id: int,
) -> HierarchyGraph:
    descendant_query = descendant_adjacency_query(concept_id)
    with get_session() as session:
        descendants = session.execute(descendant_query).fetchall()
    adjacency_list = [
        HierarchyEdge(
            descendant_concept_id=rel.descendant_concept_id,
            ancestor_concept_id=rel.ancestor_concept_id,
        )
        for rel in descendants
    ]
    concept_set = set()
    for edge in adjacency_list:
        concept_set.add(edge.ancestor_concept_id)
        concept_set.add(edge.descendant_concept_id)
    concept_query = bulk_concept_by_id_query(list(concept_set))
    with get_session() as session:
        concepts = [
                ConceptDescription.model_validate(res._mapping)
                for res in 
                session.execute(concept_query).all()
                ]
    return HierarchyGraph(concepts=concepts, adjacency_list=adjacency_list)
