from dotenv import load_dotenv
from os import environ
import pytest
from urllib.parse import quote_plus

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from omop import OMOP_match
from omop.omop_queries import (
    query_ancestors_by_name,
    query_descendants_by_name,
    query_ids_matching_name,
    query_related_by_name,
)
from utils.logging_utils import logger


# --- Testing main OMOP_match --->
@pytest.fixture
def single_query_result():
    return OMOP_match.run(
        search_term=["Acetaminophen"], logger=logger, vocabulary_id=["RxNorm"]
    )


@pytest.fixture
def three_query_result():
    return OMOP_match.run(
        search_term=["Acetaminophen", "Codeine", "Omeprazole"],
        logger=logger,
        vocabulary_id=["RxNorm"],
    )


def test_single_query_returns_one_result(single_query_result):
    assert len(single_query_result) == 1


def test_single_query_keys(single_query_result):
    assert list(single_query_result[0].keys()) == ["search_term", "CONCEPT"]


def test_single_query_concept_keys(single_query_result):
    concept_keys = list(single_query_result[0]["CONCEPT"][0])
    assert concept_keys == [
        "concept_name",
        "concept_id",
        "vocabulary_id",
        "concept_code",
        "concept_name_similarity_score",
        "CONCEPT_SYNONYM",
        "CONCEPT_ANCESTOR",
        "CONCEPT_RELATIONSHIP",
    ]


def test_three_query_returns_three_results(three_query_result):
    assert len(three_query_result) == 3


# --- Testing utility queries -->
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
