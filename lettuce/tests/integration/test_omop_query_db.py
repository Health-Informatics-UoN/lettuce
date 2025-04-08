import os
from os import environ
from dotenv import load_dotenv
import pytest
from urllib.parse import quote_plus
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session 
from unittest.mock import Mock
from haystack.dataclasses import Document

from omop.omop_queries import (
    query_ancestors_by_name,
    query_descendants_by_name,
    query_ids_matching_name,
    query_related_by_name,
)
from utils.logging_utils import logger


pytestmark = pytest.mark.skipif(os.getenv('SKIP_DATABASE_TESTS') == 'true', reason="Skipping database tests")


load_dotenv()


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