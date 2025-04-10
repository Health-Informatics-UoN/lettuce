import os
from os import environ
from dotenv import load_dotenv
import pytest
import pandas as pd 
from urllib.parse import quote_plus
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import Mock
from haystack.dataclasses import Document

from omop.omop_queries import (
    query_ancestors_by_name,
    query_descendants_by_name,
    query_ids_matching_name,
    query_related_by_name,
    query_ancestors_and_descendants_by_id
)
from utils.logging_utils import logger


pytestmark = pytest.mark.skipif(os.getenv('SKIP_DATABASE_TESTS') == 'true', reason="Skipping database tests")


load_dotenv()
DB_SCHEMA = environ["DB_SCHEMA"]


@pytest.fixture
def db_connection():
    DB_HOST = environ["DB_HOST"]
    DB_USER = environ["DB_USER"]
    DB_PASSWORD = quote_plus(environ["DB_PASSWORD"])
    DB_NAME = environ["DB_NAME"]
    DB_PORT = environ["DB_PORT"]

    connection_string = (
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    return create_engine(connection_string)


def test_matching_by_name(db_connection):
    Session = sessionmaker(db_connection)
    session = Session()

    query = query_ids_matching_name("acetaminophen", vocabulary_ids=["RxNorm"])
    results = session.execute(query).fetchall()
    session.close()

    assert len(results) == 1
    # This is the concept_id for RxNorm's acetaminophen entry - change if the underlying vocabulary changes
    assert results[0][0] == 1125315


def test_fetch_related_concepts_by_name(db_connection):
    # I'm sure there's a neater pytest-y way to keep the session active, but honestly it's not worth the bother
    Session = sessionmaker(db_connection)
    session = Session()

    query = query_related_by_name("acetaminophen", vocabulary_ids=["RxNorm"])
    results = session.execute(query).fetchall()
    session.close()

    assert len(results) > 1

    names = [result[0].concept_name for result in results]
    assert "Sinutab" in names


def test_fetch_ancestor_concepts_by_name(db_connection):
    Session = sessionmaker(db_connection)
    session = Session()

    query = query_ancestors_by_name("acetaminophen", vocabulary_ids=["RxNorm"])
    results = session.execute(query).fetchall()
    session.close()

    assert len(results) > 1

    names = [result[0].concept_name for result in results]
    assert (
        "LITTLE REMEDIES NEW BABY ESSENTIALS - acetaminophen, simethicone, zinc oxide kit"
        in names
    )


def test_fetch_ancestor_concepts_by_name_with_separation_bounds(db_connection):
    Session = sessionmaker(db_connection)
    session = Session()

    query = query_ancestors_by_name(
        "acetaminophen",
        vocabulary_ids=["RxNorm"],
        min_separation_bound=1,
        max_separation_bound=1,
    )
    results = session.execute(query).fetchall()
    session.close()

    assert len(results) > 1

    names = [result[0].concept_name for result in results]
    assert "homatropine methylbromide; systemic" in names


def test_fetch_descendant_concepts_by_name(db_connection):
    Session = sessionmaker(db_connection)
    session = Session()

    query = query_descendants_by_name("acetaminophen", vocabulary_ids=["RxNorm"])
    results = session.execute(query).fetchall()
    session.close()

    assert len(results) > 1

    names = [result[0].concept_name for result in results]
    assert (
        "Acetaminophen 0.0934 MG/MG / Ascorbic Acid 0.0333 MG/MG / Pheniramine 0.00333 MG/MG Oral Granules [FERVEX RHUME PARACETAMOL/VITAMINE C] Box of 8"
        in names
    )


def test_fetch_descendant_concepts_by_name_with_separation_bounds(db_connection):
    Session = sessionmaker(db_connection)
    session = Session()

    query = query_descendants_by_name("acetaminophen", vocabulary_ids=["RxNorm"])
    results = session.execute(query).fetchall()
    session.close()

    assert len(results) > 1

    names = [result[0].concept_name for result in results]
    assert "Painaid BRF Oral Product" in names


def test_query_descendants_and_ancestors_by_id(db_connection): 
    concept_id = 1125315  # Acetaminophen

    Session = sessionmaker(db_connection)
    session = Session()

    query = query_ancestors_and_descendants_by_id(concept_id)
    results = session.execute(query).fetchall()
    session.close()

    assert len(results) > 1
    
    columns = [
        'relationship_type', 'concept_id', 'ancestor_concept_id',
        'descendant_concept_id', 'concept_name', 'vocabulary_id',
        'concept_code', 'min_levels_of_separation', 'max_levels_of_separation'
    ]
    results = pd.DataFrame(results, columns=columns)

    names = results["concept_name"].to_list()
    assert results["relationship_type"].unique().tolist() == ["Ancestor", "Descendant"]
    assert "Painaid BRF Oral Product" in names
    assert any("analgesic" in name or "pain" in name for name in names)


def test_regression_query_descendants_and_ancestors(db_connection): 
    query = f"""
            (
                SELECT
                    'Ancestor' as relationship_type,
                    ca.ancestor_concept_id AS concept_id,
                    ca.ancestor_concept_id,
                    ca.descendant_concept_id,
                    c.concept_name,
                    c.vocabulary_id,
                    c.concept_code,
                    ca.min_levels_of_separation,
                    ca.max_levels_of_separation
                FROM
                    {DB_SCHEMA}.concept_ancestor ca
                JOIN
                    {DB_SCHEMA}.concept c ON ca.ancestor_concept_id = c.concept_id
                WHERE
                    ca.descendant_concept_id = %s AND
                    ca.min_levels_of_separation >= %s AND
                    ca.max_levels_of_separation <= %s
            )
            UNION
            (
                SELECT
                    'Descendant' as relationship_type,
                    ca.descendant_concept_id AS concept_id,
                    ca.ancestor_concept_id,
                    ca.descendant_concept_id,
                    c.concept_name,
                    c.vocabulary_id,
                    c.concept_code,
                    ca.min_levels_of_separation,
                    ca.max_levels_of_separation
                FROM
                    {DB_SCHEMA}.concept_ancestor ca
                JOIN
                    {DB_SCHEMA}.concept c ON ca.descendant_concept_id = c.concept_id
                WHERE
                    ca.ancestor_concept_id = %s AND
                    ca.min_levels_of_separation >= %s AND
                    ca.max_levels_of_separation <= %s
            )
        """
    concept_id = 1125315
    min_separation_ancestor = 1
    min_separation_descendant = 1
    max_separation_ancestor = 1
    max_separation_descendant = 1


    params = (
        concept_id,
        min_separation_ancestor,
        max_separation_ancestor,
        concept_id,
        min_separation_descendant,
        max_separation_descendant,
    )
    results_original = (
        pd.read_sql(query, con=db_connection, params=params)
        .drop_duplicates()
        .query("concept_id != @concept_id")
    )

    # New query using SQLAlchemy
    Session = sessionmaker(db_connection)
    session = Session()

    query = query_ancestors_and_descendants_by_id(
        concept_id,
        min_separation_ancestor = 1, 
        min_separation_descendant = 1, 
        max_separation_ancestor = 1, 
        max_separation_descendant = 1 
    )
    results = session.execute(query).fetchall()
    session.close()

    columns = [
        'relationship_type', 'concept_id', 'ancestor_concept_id',
        'descendant_concept_id', 'concept_name', 'vocabulary_id',
        'concept_code', 'min_levels_of_separation', 'max_levels_of_separation'
    ]
    results_refactor = pd.DataFrame(results, columns=columns)
    
    assert results_refactor.equals(results_original)