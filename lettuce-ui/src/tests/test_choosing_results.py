import pytest

from suggestions import AcceptedSuggestion, accept_suggestion, ConceptSuggestion, SuggestionRecord
from ui_utils import choose_result

@pytest.fixture
def valid_selection():
    return ConceptSuggestion(
            concept_id=4323688,
            concept_name="Cough at rest",
            domain_id="Condition",
            vocabulary_id="SNOMED",
            standard_concept="S",
            score=0.99,
            )

@pytest.fixture
def invalid_selection():
    return ConceptSuggestion(
            concept_id=-1,
            concept_name="This isn't a real concept",
            domain_id="Condition",
            vocabulary_id="SNOMED",
            standard_concept="S",
            score=0.99,
            )

@pytest.fixture
def record(valid_selection):
    return SuggestionRecord(
            search_term="coughing",
            domains=[],
            vocabs=[],
            standard_concept=True,
            valid_concept=True,
            search_mode="text-search",
            suggestion=[valid_selection]
            )

@pytest.fixture
def desired_accepted():
    return AcceptedSuggestion(
            search_term="coughing",
            domains=[],
            vocabs=[],
            search_standard_concept=True,
            valid_concept=True,
            search_mode="text-search",
            concept_id=4323688,
            concept_name="Cough at rest",
            domain_id="Condition",
            vocabulary_id="SNOMED",
            standard_concept="S",
            score=0.99,
            )

def test_accepting_valid_suggestion(valid_selection, record, desired_accepted):
    result = accept_suggestion(record, valid_selection)

    assert isinstance(result, AcceptedSuggestion)
    assert result == desired_accepted

def test_accepting_invalid_suggestion_fails(invalid_selection, record):
    with pytest.raises(KeyError):
        accept_suggestion(record, invalid_selection)
