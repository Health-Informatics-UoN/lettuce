import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from routers import search_routes

app = FastAPI()
app.include_router(search_routes.router)

client = TestClient(app)


class MockResult:
    """Mock database result object"""
    def __init__(self, concept_name, concept_id, concept_code, domain_id, vocabulary_id, 
                 concept_class_id, standard_concept, invalid_reason, ts_rank):
        self.concept_name = concept_name
        self.concept_id = concept_id
        self.concept_code = concept_code
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
            concept_code="somecode",
            vocabulary_id="SNOMED",
            concept_class_id="Clinical Finding",
            standard_concept="S",
            invalid_reason=None,
            ts_rank=0.95
        ),
        MockResult(
            concept_name="Type 2 diabetes",
            concept_id=12346,  # Changed from string to int to match Suggestion model
            domain_id="Condition",
            concept_code="anothercode",
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
        assert "items" in data
        assert "metadata" in data
        assert data["metadata"]["pipeline"] == "Full-text search"
        assert data["metadata"]["assistant"] == "Lettuce"
        assert data["metadata"]["version"] == "0.1.0"
        
        # Check items
        assert len(data["items"]) == 2
        first_recommendation = data["items"][0]
        assert first_recommendation["conceptName"] == "Diabetes mellitus"
        assert first_recommendation["conceptId"] == 12345
        assert first_recommendation["ranks"]["text_search"] == 1
        assert first_recommendation["scores"]["text_search"] == 0.95
        
        # Verify function calls - updated with new default values
        mock_ts_rank_query.assert_called_once_with(
            search_term="diabetes",
            vocabulary_id=None,
            domain_id=None,
            standard_concept=True,  # Changed from False to True
            valid_concept=True,     # Changed from False to True
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
        
        # Make request with all parameters - updated parameter names
        response = client.get(
            "/text-search/diabetes",
            params={
                "vocabulary": ["SNOMED", "ICD10CM"],  # Changed from vocabulary_id
                "domain": ["Condition", "Drug"],      # Changed from domain_id
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
        assert len(data["items"]) == 0
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
            standard_concept=True,  # Updated default value
            valid_concept=True,     # Updated default value
            top_k=5
        )
    
    @patch('routers.search_routes.get_session')
    @patch('routers.search_routes.ts_rank_query')
    def test_text_search_single_vocabulary_id(self, mock_ts_rank_query, mock_get_session, mock_db_results):
        """Test text search with single vocabulary parameter"""
        # Setup mocks
        mock_session = Mock()
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_session)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_get_session.return_value = mock_context_manager
        
        mock_session.execute.return_value.fetchall.return_value = mock_db_results
        mock_query = Mock()
        mock_ts_rank_query.return_value = mock_query
        
        # Make request with single vocabulary parameter - updated parameter name
        response = client.get("/text-search/diabetes?vocabulary=SNOMED")
        
        # Assertions
        assert response.status_code == 200
        mock_ts_rank_query.assert_called_once_with(
            search_term="diabetes",
            vocabulary_id=["SNOMED"],
            domain_id=None,
            standard_concept=True,  # Updated default value
            valid_concept=True,     # Updated default value
            top_k=5
        )
    
    @patch('routers.search_routes.get_session')
    @patch('routers.search_routes.ts_rank_query')
    def test_text_search_ranking_order(self, mock_ts_rank_query, mock_get_session):
        """Test that results are properly ranked starting from 1"""
        # Setup mocks with multiple results
        results = [
            MockResult("Concept A", 1, "codeA", "Domain", "Vocab", "Class", "S", None, 0.9),
            MockResult("Concept B", 2, "codeB", "Domain", "Vocab", "Class", "S", None, 0.8),
            MockResult("Concept C", 3, "codeC", "Domain", "Vocab", "Class", "S", None, 0.7),
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
        items = data["items"]
        assert len(items) == 3
        assert items[0]["ranks"]["text_search"] == 1
        assert items[1]["ranks"]["text_search"] == 2
        assert items[2]["ranks"]["text_search"] == 3
        
        # Check scores are preserved
        assert items[0]["scores"]["text_search"] == 0.9
        assert items[1]["scores"]["text_search"] == 0.8
        assert items[2]["scores"]["text_search"] == 0.7
    
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
        # Should return 422 validation error due to ge=1 constraint
        response = client.get("/text-search/diabetes?top_k=-1")
        assert response.status_code == 422
    
    def test_zero_top_k_parameter(self):
        """Test with zero top_k parameter"""  
        # Should return 422 validation error due to ge=1 constraint
        response = client.get("/text-search/diabetes?top_k=0")
        assert response.status_code == 422
    
    @patch('routers.search_routes.get_session')
    @patch('routers.search_routes.ts_rank_query')
    def test_text_search_with_false_flags(self, mock_ts_rank_query, mock_get_session, mock_db_results):
        """Test text search with standard_concept and valid_concept set to False"""
        # Setup mocks
        mock_session = Mock()
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_session)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_get_session.return_value = mock_context_manager
        
        mock_session.execute.return_value.fetchall.return_value = mock_db_results
        mock_query = Mock()
        mock_ts_rank_query.return_value = mock_query
        
        # Make request with flags set to False
        response = client.get(
            "/text-search/diabetes",
            params={
                "standard_concept": False,
                "valid_concept": False
            }
        )
        
        # Assertions
        assert response.status_code == 200
        mock_ts_rank_query.assert_called_once_with(
            search_term="diabetes",
            vocabulary_id=None,
            domain_id=None,
            standard_concept=False,
            valid_concept=False,
            top_k=5
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
