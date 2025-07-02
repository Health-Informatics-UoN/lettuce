import os
from unittest.mock import Mock, patch 

import pytest
from haystack.dataclasses import Document
from sqlalchemy.orm import Session

pytestmark = pytest.mark.skipif(
    os.getenv("SKIP_DATABASE_TESTS") == "true", reason="Skipping database tests"
)

from components.embeddings import PGVectorQuery


def mock_session(mock_results): 
    mock_session = Mock(spec=Session)

    # Configure mock session to return the prepared results
    mock_execute = Mock()
    mock_execute.mappings.return_value.all.return_value = mock_results
    mock_session.execute.return_value = mock_execute

    return mock_session


@pytest.fixture
def mock_session_with_valid_results(): 
    with patch("components.embeddings.get_session") as mock_get_session: 
        # Prepare mock data
        mock_results = [
            {"id": "concept1", "content": "Sample Concept 1", "score": 0.1},
            {"id": "concept2", "content": "Sample Concept 2", "score": 0.2},
        ]

        # Mimic the context manager behaviour 
        mock_get_session.return_value.__enter__.return_value = mock_session(mock_results)

        yield mock_get_session


@pytest.fixture
def mock_session_with_empty_results(): 
    with patch("components.embeddings.get_session") as mock_get_session: 
        # Prepare mock data
        mock_results = []

        # Mimic the context manager behaviour 
        mock_get_session.return_value.__enter__.return_value = mock_session(mock_results)

        yield mock_get_session


class TestPGVectorQuery:
    def test_run_with_valid_embedding(self, mock_session_with_valid_results):
        """
        Test the run method with a valid query embedding
        """
        # Create the component
        query_component = PGVectorQuery(top_k=2)

        # Run the method
        query_embedding = [0.1, 0.2, 0.3]  # Example embedding
        result = query_component.run(query_embedding=query_embedding)

        # Assertions
        assert len(result["documents"]) == 2
        assert all(isinstance(doc, Document) for doc in result["documents"])
        assert result["documents"][0].id == "concept1"
        assert result["documents"][0].content == "Sample Concept 1"
        assert result["documents"][0].score == 0.1

    def test_run_with_empty_results(self, mock_session_with_empty_results):
        """
        Test the run method when no results are found
        """
        mock_session = Mock(spec=Session)
        mock_execute = Mock()
        mock_execute.mappings.return_value.all.return_value = []
        mock_session.execute.return_value = mock_execute

        query_component = PGVectorQuery()
        query_embedding = [0.1, 0.2, 0.3]

        result = query_component.run(query_embedding=query_embedding)

        assert result["documents"] == []

    def test_run_with_top_k_parameter(self):
        """
        Test that the top_k parameter limits the number of results
        """
        query_embedding = [0.1, 0.2, 0.3]

        for k in [1, 3, 5]:
            mock_results = [
                {"id": f"concept{i}", "content": f"Sample Concept {i}", "score": 0.1 * i}
                for i in range(1, 6)
            ]
        
        with patch("components.embeddings.get_session") as mock_get_session:
            # Create the mock session
            mock_session = Mock(spec=Session)

            # Side effect that checks the limit clause
            def execute_side_effect(query):
                assert query._limit_clause.value == k, f"Expected LIMIT {k}, got {query._limit_clause.value}"
                mock_execute = Mock()
                mock_execute.mappings.return_value.all.return_value = mock_results[:k]
                return mock_execute

            mock_session.execute.side_effect = execute_side_effect
            mock_get_session.return_value.__enter__.return_value = mock_session

            # Run the component with current top_k
            query_component = PGVectorQuery(top_k=k)
            result = query_component.run(query_embedding=query_embedding)

            # Validate that only `k` documents were returned
            assert len(result["documents"]) == k


    def test_invalid_embedding_type(self, mock_session_with_valid_results):
        """
        Test handling of invalid embedding types
        """
        query_component = PGVectorQuery()

        with pytest.raises(TypeError):
            query_component.run(query_embedding="not a list")

        with pytest.raises(TypeError):
            query_component.run(query_embedding=[1, "invalid", 3])
