import re 
from collections import namedtuple
from unittest.mock import Mock, MagicMock
import pytest

from omop.omop_match import OMOPMatcher 


@pytest.fixture 
def mock_omop_matcher(mocker): 
    matcher = OMOPMatcher(
        logger=Mock(),
        vocabulary_id=["SNOMED"],
        concept_ancestor=True,
        concept_relationship=True,
        concept_synonym=True,
        search_threshold=50,
        max_separation_descendant=2,
        max_separation_ancestor=2
    )
    # Patch internal methods
    mocker.patch.object(matcher, "fetch_concept_ancestors_and_descendants", return_value=[{"mock": "ancestor"}])
    mocker.patch.object(matcher, "fetch_concept_relationships", return_value=[{"mock": "relationship"}])
    return matcher


@pytest.fixture
def mock_session(mocker):
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session  
    mock_session.__exit__.return_value = None
    mocker.patch("omop.omop_match.get_session", return_value=mock_session)
    return mock_session


class TestCalculateSimilarityScore: 
    def test_exact_match(self):
        term = "acetaminophen"
        concept_name = "acetaminophen"
        score = OMOPMatcher.calculate_similarity_score(concept_name, term)
        assert score == 100

    def test_partial_match(self):
        term = "acetaminophen"
        concept_name = "acetaminophen 10mg"
        score = OMOPMatcher.calculate_similarity_score(concept_name, term)
        assert score > 50  
    
    def test_irrelevant_match(self): 
        term = "paracetamol"
        concept_name = "skidiving"
        score = OMOPMatcher.calculate_similarity_score(concept_name, term)
        assert score < 50 

    def test_verbose_concept_name(self): 
        raw_concept = "{1 (acetaminophen 325 MG / dextromethorphan hydrobromide 10 MG / doxylamine succinate 6.25 MG Oral Capsule) / 1 (acetaminophen 325 MG / dextromethorphan hydrobromide 10 MG / phenylephrine hydrochloride 5 MG Oral Capsule) } Pack"
        term = "acetaminophen"
        cleaned_concept = re.sub(r"\(.*?\)", "", raw_concept).strip()
        raw_score = OMOPMatcher.calculate_similarity_score(raw_concept, term)
        cleaned_score = OMOPMatcher.calculate_similarity_score(cleaned_concept, term) 
    
    def test_empty_strings(self):
        score = OMOPMatcher.calculate_similarity_score("", "")
        assert score == 100

    def test_one_empty_string(self):
        score = OMOPMatcher.calculate_similarity_score("acetaminophen", "")
        assert score == 0


def test_fetch_omop_concepts_basic_case(mock_omop_matcher, mock_session):
    # Fake DB rows (matches expected output format of session.execute().fetchall())
    MockRow = namedtuple('Row', [
        "concept_id", 
        "concept_name", 
        "vocabulary_id", 
        "concept_code", 
        "concept_synonym_name"
    ]) 
    mock_data = [
        MockRow("123", "Hypertension", "SNOMED", "H123", "Hypertension Synonym"),
        MockRow("124", "Hypotension", "SNOMED", "H124", "Low BP"),
    ]

    mock_result = MagicMock()
    mock_result.fetchall.return_value = mock_data
    mock_session.execute.return_value = mock_result

    result = mock_omop_matcher.fetch_omop_concepts(search_term="Hypertension")

    assert isinstance(result, list)
    assert len(result) > 0
    assert result[0]["concept_id"] == "123"
    assert "CONCEPT_ANCESTOR" in result[0]
    assert "CONCEPT_RELATIONSHIP" in result[0]
