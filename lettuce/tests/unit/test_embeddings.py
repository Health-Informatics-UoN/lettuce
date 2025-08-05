import os
from unittest.mock import Mock

import pytest
from haystack.dataclasses import Document
from sqlalchemy.orm import Session
from components.embeddings import PGVectorQuery

pytestmark = pytest.mark.skipif(
    os.getenv("SKIP_DATABASE_TESTS") == "true", reason="Skipping database tests"
)


class TestPGVectorQuery:
    def test_initialization(self):
        """
        Test that the component can be correctly initialized with a database session
        """
        mock_session = Mock(spec=Session)
        query_component = PGVectorQuery(connection=mock_session)

        assert query_component._connection == mock_session

    def test_run_with_valid_embedding(self):
        """
        Test the run method with a valid query embedding
        """
        # Prepare mock data
        mock_session = Mock(spec=Session)
        mock_results = [
            {"id": "concept1", "content": "Sample Concept 1", "score": 0.1},
            {"id": "concept2", "content": "Sample Concept 2", "score": 0.2},
        ]

        # Configure mock session to return the prepared results
        mock_execute = Mock()
        mock_execute.mappings.return_value.all.return_value = mock_results
        mock_session.execute.return_value = mock_execute

        # Create the component
        query_component = PGVectorQuery(connection=mock_session, top_k=2)

        # Run the method
        query_embedding = [0.1, 0.2, 0.3]  # Example embedding
        result = query_component.run(query_embedding=query_embedding)

        # Assertions
        assert len(result["documents"]) == 2
        assert all(isinstance(doc, Document) for doc in result["documents"])
        assert result["documents"][0].id == "concept1"
        assert result["documents"][0].content == "Sample Concept 1"
        assert result["documents"][0].score == 0.1

    def test_run_with_empty_results(self):
        """
        Test the run method when no results are found
        """
        mock_session = Mock(spec=Session)
        mock_execute = Mock()
        mock_execute.mappings.return_value.all.return_value = []
        mock_session.execute.return_value = mock_execute

        query_component = PGVectorQuery(connection=mock_session)
        query_embedding = [0.1, 0.2, 0.3]

        result = query_component.run(query_embedding=query_embedding)

        assert result["documents"] == []

    def test_run_with_top_k_parameter(self):
        """
        Test that the top_k parameter limits the number of results
        """
        mock_session = Mock(spec=Session)
        mock_results = [
            {"id": f"concept{i}", "content": f"Sample Concept {i}", "score": 0.1 * i}
            for i in range(1, 6)
        ]

        mock_execute = Mock()
        mock_execute.mappings.return_value.all.return_value = mock_results
        mock_session.execute.return_value = mock_execute

        query_embedding = [0.1, 0.2, 0.3]

        # Test with different top_k values
        for k in [1, 3, 5]:
            # Mock the results to return only k items
            mock_execute.mappings.return_value.all.return_value = mock_results[:k]
            
            query_component = PGVectorQuery(connection=mock_session, top_k=k)
            result = query_component.run(query_embedding=query_embedding)
            assert len(result["documents"]) == k

    def test_invalid_embedding_type(self):
        """
        Test handling of invalid embedding types
        """
        mock_session = Mock(spec=Session)
        query_component = PGVectorQuery(connection=mock_session)

        with pytest.raises(TypeError):
            query_component.run(query_embedding="not a list")

        with pytest.raises(TypeError):
            query_component.run(query_embedding=[1, "invalid", 3])
