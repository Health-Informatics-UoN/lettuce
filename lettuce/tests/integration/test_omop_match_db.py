import os
import pytest

from omop.omop_match import OMOPConcept, OMOPMatcher, SearchResult 
from utils.logging_utils import logger

pytestmark = pytest.mark.skipif(os.getenv('SKIP_DATABASE_TESTS') == 'true', reason="Skipping database tests")



@pytest.fixture
def single_query_result():
    return OMOPMatcher(
        logger, 
        vocabulary_id=["RxNorm"], 
        concept_ancestor=True, 
        concept_synonym=True, 
        concept_relationship=True
    ).run(search_terms=["Acetaminophen"])


@pytest.fixture
def three_query_result():
    return OMOPMatcher(
        logger, 
        vocabulary_id=["RxNorm"], 
        concept_ancestor=True, 
        concept_synonym=True, 
        concept_relationship=True
    ).run(search_terms=["Acetaminophen", "Codeine", "Omeprazole"])


def test_single_query_returns_one_result(single_query_result):
    assert len(single_query_result) == 1


def test_single_query_concept_is_search_result(single_query_result):
    assert isinstance(single_query_result[0], SearchResult)


def test_single_query_concept_is_omop_concept(single_query_result):
    concept = single_query_result[0].concept[0]
    assert isinstance(concept, OMOPConcept)

def test_three_query_returns_three_results(three_query_result):
    assert len(three_query_result) == 3
