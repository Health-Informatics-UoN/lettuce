import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from api_models.responses import ConceptSuggestionResponse, SuggestionsMetaData, Suggestion
from routers import search_routes

app = FastAPI()
app.include_router(search_routes.router)

client = TestClient(app)


class MockResult:
    """Mock database result object"""
    def __init__(self, concept_name, concept_id, domain_id, vocabulary_id, 
                 concept_class_id, standard_concept, invalid_reason, ts_rank):
        self.concept_name = concept_name
        self.concept_id = concept_id
        self.domain_id = domain_id
        self.vocabulary_id = vocabulary_id
        self.concept_class_id = concept_class_id
        self.standard_concept = standard_concept
        self.invalid_reason = invalid_reason
        self.ts_rank = ts_rank


@pytest.fixture
def mock_db_results():
    """Sample database results for testing"""
    return [
        MockResult(
            concept_name="Diabetes mellitus",
            concept_id=12345,
            domain_id="Condition",
            vocabulary_id="SNOMED",
            concept_class_id="Clinical Finding",
            standard_concept="S",
            invalid_reason=None,
            ts_rank=0.95
        ),
        MockResult(
            concept_name="Type 2 diabetes",
            concept_id="12346",
            domain_id="Condition",
            vocabulary_id="SNOMED",
            concept_class_id="Clinical Finding",
            standard_concept="S",
            invalid_reason=None,
            ts_rank=0.85
        )
    ]


@pytest.fixture
def mock_empty_results():
    """Empty database results for testing"""
    return []


class TestTextSearchEndpoint:
    """Test suite for the text_search endpoint"""
    
    @patch('routers.search_routes.get_session')
    @patch('routers.search_routes.ts_rank_query')
    def test_basic_text_search(self, mock_ts_rank_query, mock_get_session, mock_db_results):
        """Test basic text search functionality"""
        # Setup mocks
        mock_session = Mock()
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_session)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_get_session.return_value = mock_context_manager
        
        mock_session.execute.return_value.fetchall.return_value = mock_db_results
        mock_query = Mock()
        mock_ts_rank_query.return_value = mock_query
        
        # Make request
        response = client.get("/text-search/diabetes")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "recommendations" in data
        assert "metadata" in data
        assert data["metadata"]["pipeline"] == "Full-text search"
        
        # Check recommendations
        assert len(data["recommendations"]) == 2
        first_recommendation = data["recommendations"][0]
        assert first_recommendation["concept_name"] == "Diabetes mellitus"
        assert first_recommendation["concept_id"] == 12345
        assert first_recommendation["ranks"]["text_search"] == 1
        assert first_recommendation["scores"]["text_search"] == 0.95
        
        # Verify function calls
        mock_ts_rank_query.assert_called_once_with(
            search_term="diabetes",
            vocabulary_id=None,
            domain_id=None,
            standard_concept=False,
            valid_concept=False,
            top_k=5
        )
        mock_session.execute.assert_called_once_with(mock_query)
    
    @patch('routers.search_routes.get_session')
    @patch('routers.search_routes.ts_rank_query')
    def test_text_search_with_all_parameters(self, mock_ts_rank_query, mock_get_session, mock_db_results):
        """Test text search with all query parameters"""
        # Setup mocks
        mock_session = Mock()
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_session)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_get_session.return_value = mock_context_manager
        
        mock_session.execute.return_value.fetchall.return_value = mock_db_results
        mock_query = Mock()
        mock_ts_rank_query.return_value = mock_query
        
        # Make request with all parameters
        response = client.get(
            "/text-search/diabetes",
            params={
                "vocabulary_id": ["SNOMED", "ICD10CM"],
                "domain_id": ["Condition", "Drug"],
                "standard_concept": True,
                "valid_concept": True,
                "top_k": 10
            }
        )
        
        # Assertions
        assert response.status_code == 200
        
        # Verify function was called with correct parameters
        mock_ts_rank_query.assert_called_once_with(
            search_term="diabetes",
            vocabulary_id=["SNOMED", "ICD10CM"],
            domain_id=["Condition", "Drug"],
            standard_concept=True,
            valid_concept=True,
            top_k=10
        )
    
    @patch('routers.search_routes.get_session')
    @patch('routers.search_routes.ts_rank_query')
    def test_text_search_empty_results(self, mock_ts_rank_query, mock_get_session, mock_empty_results):
        """Test text search with no results"""
        # Setup mocks
        mock_session = Mock()
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_session)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_get_session.return_value = mock_context_manager
        
        mock_session.execute.return_value.fetchall.return_value = mock_empty_results
        mock_query = Mock()
        mock_ts_rank_query.return_value = mock_query
        
        # Make request
        response = client.get("/text-search/nonexistent")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert len(data["recommendations"]) == 0
        assert data["metadata"]["pipeline"] == "Full-text search"
    
    @patch('routers.search_routes.get_session')
    @patch('routers.search_routes.ts_rank_query')
    def test_text_search_special_characters(self, mock_ts_rank_query, mock_get_session, mock_db_results):
        """Test text search with special characters in search term"""
        # Setup mocks
        mock_session = Mock()
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_session)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_get_session.return_value = mock_context_manager
        
        mock_session.execute.return_value.fetchall.return_value = mock_db_results
        mock_query = Mock()
        mock_ts_rank_query.return_value = mock_query
        
        # Make request with special characters
        search_term = "diabetes type-2 (adult)"
        response = client.get(f"/text-search/{search_term}")
        
        # Assertions
        assert response.status_code == 200
        mock_ts_rank_query.assert_called_once_with(
            search_term=search_term,
            vocabulary_id=None,
            domain_id=None,
            standard_concept=False,
            valid_concept=False,
            top_k=5
        )
    
    @patch('routers.search_routes.get_session')
    @patch('routers.search_routes.ts_rank_query')
    def test_text_search_single_vocabulary_id(self, mock_ts_rank_query, mock_get_session, mock_db_results):
        """Test text search with single vocabulary_id parameter"""
        # Setup mocks
        mock_session = Mock()
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_session)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_get_session.return_value = mock_context_manager
        
        mock_session.execute.return_value.fetchall.return_value = mock_db_results
        mock_query = Mock()
        mock_ts_rank_query.return_value = mock_query
        
        # Make request with single vocabulary_id
        response = client.get("/text-search/diabetes?vocabulary_id=SNOMED")
        
        # Assertions
        assert response.status_code == 200
        mock_ts_rank_query.assert_called_once_with(
            search_term="diabetes",
            vocabulary_id=["SNOMED"],
            domain_id=None,
            standard_concept=False,
            valid_concept=False,
            top_k=5
        )
    
    @patch('routers.search_routes.get_session')
    @patch('routers.search_routes.ts_rank_query')
    def test_text_search_ranking_order(self, mock_ts_rank_query, mock_get_session):
        """Test that results are properly ranked starting from 1"""
        # Setup mocks with multiple results
        results = [
            MockResult("Concept A", "1", "Domain", "Vocab", "Class", "S", None, 0.9),
            MockResult("Concept B", "2", "Domain", "Vocab", "Class", "S", None, 0.8),
            MockResult("Concept C", "3", "Domain", "Vocab", "Class", "S", None, 0.7),
        ]
        
        mock_session = Mock()
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_session)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_get_session.return_value = mock_context_manager
        
        mock_session.execute.return_value.fetchall.return_value = results
        mock_query = Mock()
        mock_ts_rank_query.return_value = mock_query
        
        # Make request
        response = client.get("/text-search/test")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        # Check ranking order
        recommendations = data["recommendations"]
        assert len(recommendations) == 3
        assert recommendations[0]["ranks"]["text_search"] == 1
        assert recommendations[1]["ranks"]["text_search"] == 2
        assert recommendations[2]["ranks"]["text_search"] == 3
        
        # Check scores are preserved
        assert recommendations[0]["scores"]["text_search"] == 0.9
        assert recommendations[1]["scores"]["text_search"] == 0.8
        assert recommendations[2]["scores"]["text_search"] == 0.7
    
    @patch('routers.search_routes.get_session')
    @patch('routers.search_routes.ts_rank_query')
    def test_database_exception_handling(self, mock_ts_rank_query, mock_get_session):
        """Test handling of database exceptions"""
        # Setup mocks to raise exception
        mock_session = Mock()
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_session)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_get_session.return_value = mock_context_manager
        
        mock_session.execute.side_effect = Exception("Database connection error")
        mock_query = Mock()
        mock_ts_rank_query.return_value = mock_query
        
        # Make request - should raise exception (or handle gracefully depending on your error handling)
        with pytest.raises(Exception):
            client.get("/text-search/diabetes")
    
    def test_invalid_top_k_parameter(self):
        """Test with invalid top_k parameter"""
        response = client.get("/text-search/diabetes?top_k=invalid")
        assert response.status_code == 422  # Validation error
    
    def test_negative_top_k_parameter(self):
        """Test with negative top_k parameter"""
        # This test depends on whether you have validation for top_k >= 0
        response = client.get("/text-search/diabetes?top_k=-1")
        # Should either return 422 (validation error) or handle gracefully
        assert response.status_code in [200, 422]
    
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
