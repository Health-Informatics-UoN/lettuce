import os

import pytest 
from fastapi.testclient import TestClient
from unittest.mock import patch
from api_models.responses import Suggestion, SuggestionsMetaData, ConceptSuggestionResponse

from api import app 


client = TestClient(app)


@pytest.fixture
def mock_text_search():
    def simple_mock(request):
        meta = SuggestionsMetaData()
        suggestion = Suggestion(
                conceptName="Cough at rest",
                conceptId=4323688,
                conceptCode="7142008",
                domain="Condition",
                vocabulary="SNOMED",
                conceptClass="Clinical finding",
                standard_concept="S",
                invalid_reason=None,
                ranks=None,
                scores=None,
                )
        return ConceptSuggestionResponse(items=[suggestion], metadata=meta)
    
    with patch('routers.search_routes.text_search', side_effect=simple_mock) as mock:
        yield mock


class TestAuthentication:
    
    def test_missing_authorization_header(self, mock_text_search):
        """Test request without Authorization header"""
        response = client.get(
            "/search/text-search/coughing", 
        )
        
        assert response.status_code in [401, 403]  
    
    def test_invalid_authorization_format(self, mock_text_search):
        """Test malformed Authorization header"""
        headers = {"Authorization": "InvalidFormat token123"}
        response = client.get(
            "/search/text-search/coughing", 
            headers=headers
        )
        
        assert response.status_code == 403

    @patch.dict(os.environ, {"AUTH_API_KEY": "test_key_123"}) 
    def test_invalid_api_key(self, mock_text_search):
        """Test with wrong API key"""
        headers = {"Authorization": "Bearer wrong_key"}
        response = client.get(
            "/search/text-search/coughing", 
            headers=headers
        )
        
        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]
    
    @patch.dict(os.environ, {"AUTH_API_KEY": "test_key_123"})
    def test_valid_api_key(self, mock_text_search):
        """Test with correct API key"""

        headers = {"Authorization": "Bearer test_key_123"}
        response = client.get(
            "/search/text-search/coughing", 
            headers=headers
        )
        assert response.status_code == 200
