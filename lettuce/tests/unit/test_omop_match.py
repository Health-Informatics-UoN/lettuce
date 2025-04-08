import os
from collections import namedtuple
from unittest.mock import Mock, MagicMock
import pytest
import pandas as pd 
from haystack.dataclasses import Document
from sqlalchemy.orm import Session

from omop.OMOP_match import OMOPMatcher 


@pytest.fixture 
def mock_omop_matcher(mocker): 
    matcher = OMOPMatcher(logger=Mock())
    # Patch internal methods
    mocker.patch.object(matcher, "fetch_concept_ancestor", return_value=[{"mock": "ancestor"}])
    mocker.patch.object(matcher, "fetch_concept_relationship", return_value=[{"mock": "relationship"}])
    return matcher


@pytest.fixture
def mock_session(mocker):
    mock_session = MagicMock(spec=Session)
    mock_session_factory = MagicMock(return_value=mock_session)
    mocker.patch("omop.OMOP_match.sessionmaker", return_value=mock_session_factory)
    return mock_session


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
 
    result = mock_omop_matcher.fetch_OMOP_concepts(
        search_term="Hypertension",
        vocabulary_id=["SNOMED"],
        concept_ancestor=True,
        concept_relationship=True,
        concept_synonym=True,
        search_threshold=50,
        max_separation_descendant=2,
        max_separation_ancestor=2,
    )

    assert isinstance(result, list)
    assert len(result) > 0
    assert result[0]["concept_id"] == "123"
    assert "CONCEPT_ANCESTOR" in result[0]
    assert "CONCEPT_RELATIONSHIP" in result[0]