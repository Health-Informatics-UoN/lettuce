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
    """
    Build a query to count the total number of distinct concepts in the database.

    Returns
    -------
    Select
        SQLAlchemy Select object that returns the count of distinct concept_ids
        in the Concept table.
    """
    return select(sa.func.count(distinct(Concept.concept_id)))

def get_domains() -> Select:
    """
    Build a query to retrieve the domain IDs from the concepts in your OMOP-CDM
    database in order of the number of concepts in that domain

    Returns
    -------
    Select
        SQLAlchemy Select object that returns the distinct domain_ids in the
        concept table
    """
    return select(Concept.domain_id).group_by(Concept.domain_id).order_by(sa.func.count(distinct(Concept.concept_id)))

def get_vocabs() -> Select:
    """
    Build a query to retrieve the vocabulary IDs from the concepts in your OMOP-CDM
    database in order of the number of concepts in that vocabulary

    Returns
    -------
    Select
        SQLAlchemy Select object that returns the distinct vocabulary_ids in the
        concept table
    """
    return select(Concept.vocabulary_id).group_by(Concept.vocabulary_id).order_by(sa.func.count(distinct(Concept.concept_id)))

def ts_rank_query(
        search_term: str,
        vocabulary_id: Optional[List[str]],
        domain_id: Optional[List[str]],
        standard_concept: bool,
        valid_concept: bool,
        top_k: int,
        ) -> Select:
    """
    Build a full-text search query using PostgreSQL's ts_rank functionality.

    This function creates a query that uses PostgreSQL's full-text search capabilities
    to find concepts matching a search term, ranked by relevance using ts_rank.
    The search term is preprocessed before being converted to a tsquery.

    Parameters
    ----------
    search_term : str
        The term to search for in concept names
    vocabulary_id : List[str] | None
        List of vocabulary IDs to filter by, or None for all vocabularies
    domain_id : List[str] | None
        List of domain IDs to filter by, or None for all domains
    standard_concept : bool
        If True, only return standard concepts (standard_concept = 'S')
    valid_concept : bool
        If True, only return valid concepts (invalid_reason is None)
    top_k : int
        Maximum number of results to return

    Returns
    -------
    Select
        SQLAlchemy Select object ordered by ts_rank score (highest first)
        and limited to top_k results.
    """
    pp_search = preprocess_search_term(search_term)
    ts_query = sa.func.to_tsquery("english", pp_search)
    ts_rank_col = sa.func.ts_rank(Concept.concept_name_tsv, ts_query, 16).label("ts_rank")
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
    Builds an OMOP query to search for concepts using full-text search.

    Uses the ORM models for the concept and concept_synonym tables to build a query
    that searches concept names and optionally concept synonyms using PostgreSQL's
    full-text search capabilities.

    Parameters
    ----------
    search_term : str
        The term to use when searching the relevant tables for concepts
    vocabulary_id : list[str] | None
        A list of vocabulary_ids in the concepts table. The returned concepts 
        will have one of these vocabulary_ids, or None for all vocabularies
    standard_concept : bool
        If True, only return standard concepts (standard_concept = 'S')
    concept_synonym : bool
        If True, then the query is expanded to find matches using the concept_synonym table

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
    """
    Build a query to retrieve all distinct vocabulary IDs from the concept table.

    Returns
    -------
    Select
        SQLAlchemy Select object for retrieving all unique vocabulary_id values
        from the Concept table.
    """
    return select(Concept.vocabulary_id.distinct())


def query_ids_matching_name(
        query_concept,
        vocabulary_ids: list[str] | None,
        full_concept: bool = False
        ) -> Select:
    """
    Build a query to retrieve concept IDs that match a specified concept name.

    Performs case-insensitive matching on concept names and optionally filters
    by vocabulary IDs.

    Parameters
    ----------
    query_concept : str
        The concept name to match (case-insensitive)
    vocabulary_ids : list[str] | None
        Optional list of vocabulary IDs to filter by
    full_concept : bool, optional
        If True, return full concept details; if False, return only concept_id.
        Defaults to False.

    Returns
    -------
    Select
        SQLAlchemy Select object for the constructed query
    """
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
    """
    Build a query to find ancestors of concepts matching a specified name.

    This function finds all concepts in the hierarchy that are ancestors of
    concepts whose names match the provided query string, within specified
    hierarchical distance bounds.

    Parameters
    ----------
    query_concept : str
        The concept name to match
    vocabulary_ids : list[str] | None
        Optional list of vocabulary IDs to filter by
    min_separation_bound : int, optional
        Minimum levels of separation from the matching concept. Defaults to 0.
    max_separation_bound : int | None, optional
        Maximum levels of separation from the matching concept. Defaults to None (no limit).

    Returns
    -------
    Select
        SQLAlchemy Select object for the constructed query
    """
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
    if max_separation_bound:
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
    """
    Build a query to find descendants of concepts matching a specified name.

    This function finds all concepts in the hierarchy that are descendants of
    concepts whose names match the provided query string, within specified
    hierarchical distance bounds.

    Parameters
    ----------
    query_concept : str
        The concept name to match
    vocabulary_ids : list[str] | None
        Optional list of vocabulary IDs to filter by
    min_separation_bound : int, optional
        Minimum levels of separation from the matching concept. Defaults to 0.
    max_separation_bound : int | None, optional
        Maximum levels of separation from the matching concept. Defaults to None (no limit).

    Returns
    -------
    Select
        SQLAlchemy Select object for the constructed query
    """
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

    This function creates a union query that retrieves both ancestor and descendant
    concepts within specified hierarchical distances. If max_separation values are 
    None, they default to 1000 (essentially unlimited). The query returns a 
    relationship type ('Ancestor' or 'Descendant') to distinguish the results.
    
    Parameters
    ----------
    concept_id : int
        The concept_id to find hierarchy for
    min_separation_ancestor : int, optional
        Minimum levels of separation for ancestors. Defaults to 1.
    max_separation_ancestor : int | None, optional
        Maximum levels of separation for ancestors. Defaults to 1.
    min_separation_descendant : int, optional
        Minimum levels of separation for descendants. Defaults to 1.
    max_separation_descendant : int | None, optional
        Maximum levels of separation for descendants. Defaults to 1.
        
    Returns
    -------
    CompoundSelect
        SQLAlchemy union query combining ancestor and descendant results
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
    """
    Build a query to find concepts related to concepts matching a specified name.

    This function searches for concepts whose names match the provided query string,
    then returns concepts that are related to the matching concepts through 
    ConceptRelationship entries.

    Parameters
    ----------
    query_concept : str
        The concept name to match
    vocabulary_ids : list[str] | None
        Optional list of vocabulary IDs to filter by

    Returns
    -------
    Select
        SQLAlchemy Select object for the constructed query
    """
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
    Only returns active relationships (where valid_end_date is in the future)
    and excludes self-relationships.
    
    Parameters
    ----------
    concept_id : int
        The source concept ID for which to find related concepts
        
    Returns
    -------
    Select 
        SQLAlchemy Select object representing the query with detailed information
        about both the relationship and the related concept.
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
        describe_concept: bool = False,
        ) -> Select:
    """
    Build a query to find concepts with embeddings similar to the provided vector.

    Uses PostgreSQL vector operations for similarity calculation and orders results
    by similarity score. The function allows filtering by vocabulary, domain, 
    standard concept status, and validity.

    Parameters
    ----------
    query_embedding : vector
        The vector embedding to compare against (from a pre-trained embeddings model)
    embed_vocab : List[str] | None, optional
        Optional list of vocabulary IDs to filter by. Defaults to None.
    domain_id : List[str] | None, optional  
        Optional list of domain IDs to filter by. Defaults to None.
    standard_concept : bool, optional
        If True, only include standard concepts (standard_concept = 'S'). Defaults to False.
    valid_concept : bool, optional
        If True, only include valid concepts (invalid_reason is None). Defaults to False.
    n : int, optional
        Maximum number of results to return. Defaults to 5.
    describe_concept : bool, optional
        If True, return full concept details; if False, return minimal concept info. 
        Defaults to False.

    Returns
    -------
    Select
        SQLAlchemy Select object for the constructed query, ordered by similarity score
        (lowest distance/highest similarity first) and limited to n results.
    """
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
