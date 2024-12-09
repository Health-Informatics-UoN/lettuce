from omop.omop_models import (
    Concept,
    ConceptRelationship,
    ConceptSynonym,
    ConceptAncestor,
)

from sqlalchemy import select, or_, func
from sqlalchemy.sql import Select, text, null


def text_search_query(
    search_term: str, vocabulary_id: list[str] | None, concept_synonym: bool
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
        "to_tsvector('english', concept_name) @@ to_tsquery('english', :search_term)"
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
    ).where(Concept.standard_concept == "S")

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


def query_ids_matching_name(query_concept, vocabulary_ids: list[str] | None) -> Select:
    base_query = select(
        Concept.concept_id,
    ).where(func.lower(Concept.concept_name) == query_concept.lower())
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


def query_ancestors_by_id() -> Select: ...


def query_related_by_name(query_concept: str, vocabulary_ids: list[str]) -> Select:
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


def query_related_by_id() -> Select: ...
