from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, select, func, or_

from sqlalchemy.sql import Select, text, null

from dotenv import load_dotenv
from os import environ


load_dotenv()
DB_SCHEMA = environ["DB_SCHEMA"]

Base = declarative_base()


class Concept(Base):
    """
    This class represents an ORM mapping to the OMOP concept table
    """

    __tablename__ = "concept"
    __table_args__ = {"schema": DB_SCHEMA}

    concept_id = Column(Integer, primary_key=True)
    concept_name = Column(String)
    vocabulary_id = Column(String)
    concept_code = Column(String)
    standard_concept = Column(String)

    def __repr__(self) -> str:
        return super().__repr__()


class ConceptSynonym(Base):
    """
    This class represents an ORM mapping to the OMOP concept_synonym table
    """

    __tablename__ = "concept_synonym"
    __table_args__ = {"schema": DB_SCHEMA}

    concept_id = Column(Integer, primary_key=True)
    concept_synonym_name = Column(String)
    language_concept_id = Column(Integer)

    def __repr__(self) -> str:
        return super().__repr__()


def build_query(
    search_term: str, vocabulary_id: list[str] | None, concept_synonym: str
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

    if concept_synonym == "y":
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
