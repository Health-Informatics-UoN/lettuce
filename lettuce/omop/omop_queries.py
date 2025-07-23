from omop.omop_models import (
    Concept,
    ConceptRelationship,
    ConceptSynonym,
    ConceptAncestor,
    Embedding,
)

import sqlalchemy as sa
from sqlalchemy import select, or_, func, literal, distinct
from sqlalchemy.sql import Select, CompoundSelect, text, null
from typing import List, Optional

from omop.preprocess import preprocess_search_term

def count_concepts() -> Select:
    return select(sa.func.count(distinct(Concept.concept_id)))

def ts_rank_query(
        search_term: str,
        vocabulary_id: Optional[List[str]],
        domain_id: Optional[List[str]],
        standard_concept: bool,
        valid_concept: bool,
        top_k: int,
        ) -> Select:
    pp_search = preprocess_search_term(search_term)
    ts_query = sa.func.to_tsquery("english", pp_search)
    ts_rank_col = sa.func.ts_rank(Concept.concept_name_tsv, ts_query).label("ts_rank")
    query = select(
            Concept.concept_name,
            Concept.concept_id,
            Concept.concept_code,
            Concept.domain_id,
            Concept.vocabulary_id,
            Concept.concept_class_id,
            Concept.standard_concept,
            Concept.invalid_reason,
            ts_rank_col,
            )
    if vocabulary_id is not None:
        query = query.where(Concept.vocabulary_id.in_(vocabulary_id))
    if domain_id is not None:
        query = query.where(Concept.domain_id.in_(domain_id))
    if standard_concept:
        query = query.where(Concept.standard_concept == "S")
    if valid_concept:
        query = query.where(Concept.invalid_reason == None)

    return  query.where(
                Concept.concept_name_tsv.bool_op("@@")(ts_query)
            ).order_by(
                ts_rank_col.desc()
            ).limit(top_k)



def text_search_query(
        search_term: str, vocabulary_id: list[str] | None, standard_concept:bool, concept_synonym: bool
) -> Select:
    """
    Builds an OMOP query to search for concepts

    Uses the ORM models for the concept and concept_synonym tables to build a query

    Parameters
    ----------
    search_term: str
        The term to use when searching the relevant tables for concepts
    vocabulary_id: list[str]
        A list of vocabulary_ids in the concepts table. The returned concepts will have one of these vocabulary_ids
    concept_synonym: str
        If 'y', then the query is expanded to find matches using the concept_synonym table

    Returns
    -------
    Select
        An SQLAlchemy Select for the desired query
    """
    concept_ts_condition = text(
        "concept_name_tsv @@ to_tsquery('english', :search_term)"
    )
    synonym_ts_condition = text(
        "to_tsvector('english', concept_synonym_name) @@ to_tsquery('english', :search_term)"
    )

    # Base query
    query = select(
        Concept.concept_id,
        Concept.concept_name,
        Concept.vocabulary_id,
        Concept.concept_code,
    )
    
    if standard_concept:
        query = query.where(Concept.standard_concept == "S")

    if vocabulary_id:
        query = query.where(Concept.vocabulary_id.in_(vocabulary_id))

    if concept_synonym:
        # Define the synonym matches CTE
        synonym_matches = (
            select(
                ConceptSynonym.concept_id.label("synonym_concept_id"),
                ConceptSynonym.concept_synonym_name,
            )
            .join(Concept, ConceptSynonym.concept_id == Concept.concept_id)
            .where(Concept.standard_concept == "S")
            .where(synonym_ts_condition.bindparams(search_term=search_term))
        )
        if vocabulary_id:
            synonym_matches = synonym_matches.where(
                Concept.vocabulary_id.in_(vocabulary_id)
            )

        synonym_matches_cte = synonym_matches.cte("synonym_matches")

        # Use the CTE in the main query
        query = query.add_columns(synonym_matches_cte.c.concept_synonym_name)
        query = query.outerjoin(
            synonym_matches_cte,
            Concept.concept_id == synonym_matches_cte.c.synonym_concept_id,
        )
        query = query.where(
            or_(
                concept_ts_condition.bindparams(search_term=search_term),
                Concept.concept_id == synonym_matches_cte.c.synonym_concept_id,
            )
        )
    else:
        query = query.add_columns(null().label("concept_synonym_name"))
        query = query.where(concept_ts_condition.bindparams(search_term=search_term))

    return query


def get_all_vocabs() -> Select:
    return select(Concept.vocabulary_id.distinct())


def query_ids_matching_name(
        query_concept,
        vocabulary_ids: list[str] | None,
        full_concept: bool = False
        ) -> Select:
    if full_concept:
        base_query = select(
            Concept.concept_name,
            Concept.concept_id,
            Concept.concept_code,
            Concept.domain_id,
            Concept.vocabulary_id,
            Concept.concept_class_id,
            Concept.standard_concept,
            Concept.invalid_reason,
            )
    else:
        base_query = select(Concept.concept_id)
    base_query = base_query.where(func.lower(Concept.concept_name) == query_concept.lower())
    if vocabulary_ids:
        return base_query.where(Concept.vocabulary_id.in_(vocabulary_ids))
    else:
        return base_query


def query_ancestors_by_name(
    query_concept: str,
    vocabulary_ids: list[str] | None,
    min_separation_bound: int = 0,
    max_separation_bound: int | None = None,
) -> Select:

    matching_names = query_ids_matching_name(query_concept, vocabulary_ids).cte()
    ancestors = (
        select(Concept)
        .join(
            ConceptAncestor, ConceptAncestor.ancestor_concept_id == Concept.concept_id
        )
        .join(
            matching_names,
            ConceptAncestor.descendant_concept_id == matching_names.c.concept_id,
        )
        .where(ConceptAncestor.min_levels_of_separation >= min_separation_bound)
    )
    if min_separation_bound:
        return ancestors.where(
            ConceptAncestor.max_levels_of_separation <= max_separation_bound
        )
    else:
        return ancestors


def query_descendants_by_name(
    query_concept: str,
    vocabulary_ids: list[str] | None,
    min_separation_bound: int = 0,
    max_separation_bound: int | None = None,
) -> Select:

    matching_names = query_ids_matching_name(query_concept, vocabulary_ids).cte()
    descendants = (
        select(Concept)
        .join(
            ConceptAncestor, ConceptAncestor.descendant_concept_id == Concept.concept_id
        )
        .join(
            matching_names,
            ConceptAncestor.ancestor_concept_id == matching_names.c.concept_id,
        )
        .where(ConceptAncestor.min_levels_of_separation >= min_separation_bound)
    )
    if max_separation_bound:
        return descendants.where(
            ConceptAncestor.max_levels_of_separation <= max_separation_bound
        )
    else:
        return descendants


def query_ancestors_and_descendants_by_id(
    concept_id: int,
    min_separation_ancestor: int = 1,
    max_separation_ancestor: int | None = 1,
    min_separation_descendant: int = 1,
    max_separation_descendant: int | None = 1
) -> CompoundSelect: 
    """
    Build a query to find both ancestors and descendants of a concept. 
    
    Parameters
    ----------
    concept_id: int
        The concept_id to find hierarchy for
    min_separation_ancestor: int
        Minimum levels of separation for ancestors
    max_separation_ancestor: int
        Maximum levels of separation for ancestors
    min_separation_descendant: int
        Minimum levels of separation for descendants
    max_separation_descendant: int
        Maximum levels of separation for descendants
        
    Returns
    -------
    Select
        SQLAlchemy Select object representing the query
    """
    if max_separation_ancestor is None:
        max_separation_ancestor = 1000
    if max_separation_descendant is None: 
        max_separation_descendant = 1000 

    ancestors = (
        select(
            literal('Ancestor').label('relationship_type'),
            ConceptAncestor.ancestor_concept_id.label('concept_id'),
            ConceptAncestor.ancestor_concept_id,
            ConceptAncestor.descendant_concept_id,
            Concept.concept_name,
            Concept.vocabulary_id,
            Concept.concept_code,
            ConceptAncestor.min_levels_of_separation,
            ConceptAncestor.max_levels_of_separation
        )
        .select_from(ConceptAncestor)
        .join(
            Concept, 
            ConceptAncestor.ancestor_concept_id == Concept.concept_id
        )
        .where(
            ConceptAncestor.descendant_concept_id == concept_id,
            ConceptAncestor.min_levels_of_separation >= min_separation_ancestor,
            ConceptAncestor.max_levels_of_separation <= max_separation_ancestor
        )
    )

    descendants = (
        select(
            literal('Descendant').label('relationship_type'),
            ConceptAncestor.descendant_concept_id.label('concept_id'),
            ConceptAncestor.ancestor_concept_id,
            ConceptAncestor.descendant_concept_id,
            Concept.concept_name,
            Concept.vocabulary_id,
            Concept.concept_code,
            ConceptAncestor.min_levels_of_separation,
            ConceptAncestor.max_levels_of_separation
        )
        .select_from(ConceptAncestor)
        .join(
            Concept, 
            ConceptAncestor.descendant_concept_id == Concept.concept_id
        )
        .where(
            ConceptAncestor.ancestor_concept_id == concept_id,
            ConceptAncestor.min_levels_of_separation >= min_separation_descendant,
            ConceptAncestor.max_levels_of_separation <= max_separation_descendant
        )
    )

    return ancestors.union(descendants)


def query_related_by_name(
    query_concept: str, vocabulary_ids: list[str] | None
) -> Select:
    matching_names = query_ids_matching_name(query_concept, vocabulary_ids).cte()
    return (
        select(Concept)
        .join(
            ConceptRelationship, ConceptRelationship.concept_id_2 == Concept.concept_id
        )
        .join(
            matching_names,
            ConceptRelationship.concept_id_1 == matching_names.c.concept_id,
        )
    )


def query_related_by_id(concept_id: int) -> Select:
    """
    Build a query to find all concepts related to a given concept ID.
    
    This function creates a SQLAlchemy query that retrieves related concepts
    from the concept_relationship table of the OMOP database. It finds all
    active relationships where the given concept_id is the source concept.
    
    Parameters
    ----------
    concept_id : int
        The source concept ID for which to find related concepts
        
    Returns
    -------
    Select 
        SQLAlchemy Select object representing the query
    """
    related = (
        select(
            ConceptRelationship.concept_id_2.label("concept_id"), 
            ConceptRelationship.concept_id_1, 
            ConceptRelationship.relationship_id, 
            ConceptRelationship.concept_id_2, 
            Concept.concept_name,
            Concept.vocabulary_id,
            Concept.concept_code
        )
        .select_from(ConceptRelationship)
        .join(
            Concept, 
            ConceptRelationship.concept_id_2 == Concept.concept_id 
        )
        .where(
            (ConceptRelationship.concept_id_1 == concept_id) &
            (ConceptRelationship.valid_end_date > func.now()) & 
            (ConceptRelationship.concept_id_2 != ConceptRelationship.concept_id_1) 
        )
    )
    return related 


def query_vector(
        query_embedding,
        embed_vocab: List[str] | None = None,
        domain_id: List[str] | None = None,
        standard_concept: bool = False,
        valid_concept: bool = False,
        n: int = 5,
        describe_concept:bool = False,
        ) -> Select:
    filtered_concepts = select(Concept.concept_id)
    if embed_vocab is not None:
        filtered_concepts = filtered_concepts.where(Concept.vocabulary_id.in_(embed_vocab))
    if domain_id is not None:
        filtered_concepts = filtered_concepts.where(Concept.domain_id.in_(domain_id))
    if standard_concept:
        filtered_concepts = filtered_concepts.where(Concept.standard_concept == "S")
    if valid_concept:
        filtered_concepts = filtered_concepts.where(Concept.invalid_reason == None)

    score_expr = Embedding.score(query_embedding).label("score")
    
    embedding_scores = (
            select(Embedding.concept_id, score_expr)
            .where(Embedding.concept_id.in_(filtered_concepts))
            .order_by(score_expr)
            .limit(n)
            .cte("embedding_result")
        )


    if describe_concept:
        query = (
            select(Concept, embedding_scores.c.score)
            .join(Concept, Concept.concept_id == embedding_scores.c.concept_id)
        )
    else:
        query = (
            select(
                Concept.concept_id.label("id"),
                Concept.concept_name.label("content"),
                embedding_scores.c.score,
            )
            .join(Concept, Concept.concept_id == embedding_scores.c.concept_id)
        )
    
    return query
