import pytest
from omop import OMOP_match
from options.base_options import BaseOptions
from utils.logging_utils import Logger

@pytest.fixture
def single_query_result():
    logger = Logger().make_logger()
    opt = BaseOptions()
    opt.initialize()
    opt = opt.parse()
    return OMOP_match.run(opt=opt,
                            search_term=["Acetaminophen"],
                            logger=logger
                            )

@pytest.fixture
def three_query_result():
    logger = Logger().make_logger()
    opt = BaseOptions()
    opt.initialize()
    opt = opt.parse()
    return OMOP_match.run(
            opt=opt,
            search_term=['Acetaminophen', 'Codeine', 'Omeprazole'],
            logger=logger
            )


def test_single_query_returns_one_result(single_query_result):
    assert len(single_query_result) == 1

def test_single_query_keys(single_query_result):
    assert list(single_query_result[0].keys()) == ['search_term', 'CONCEPT']

def test_single_query_concept_keys(single_query_result):
    concept_keys = list(single_query_result[0]['CONCEPT'][0])
    assert concept_keys == ['concept_name', 'concept_id', 'vocabulary_id', 'concept_code', 'concept_name_similarity_score', 'CONCEPT_SYNONYM', 'CONCEPT_ANCESTOR', 'CONCEPT_RELATIONSHIP']

def test_three_query_returns_three_results(three_query_result):
    assert len(three_query_result) == 3
