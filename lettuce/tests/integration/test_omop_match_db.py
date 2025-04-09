import os
from os import environ
from dotenv import load_dotenv
import pytest
from urllib.parse import quote_plus
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session 
from unittest.mock import Mock
from haystack.dataclasses import Document

pytestmark = pytest.mark.skipif(os.getenv('SKIP_DATABASE_TESTS') == 'true', reason="Skipping database tests")

from omop.OMOP_match import OMOPMatcher 
from utils.logging_utils import logger


@pytest.fixture
def single_query_result():
    return OMOPMatcher(logger).run(
        search_terms=["Acetaminophen"], vocabulary_id=["RxNorm"]
    )


@pytest.fixture
def three_query_result():
    return OMOPMatcher(logger).run(
        search_terms=["Acetaminophen", "Codeine", "Omeprazole"],
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
