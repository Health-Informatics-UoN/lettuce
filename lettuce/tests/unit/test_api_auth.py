import os
import pytest 
from fastapi.testclient import TestClient
from unittest.mock import patch

from api import app 


client = TestClient(app)


@pytest.fixture
def mock_generate_events():
    async def simple_mock(request):
        yield '{"event": "test", "data": "mocked"}'
    
    with patch('routers.pipeline_routes.generate_events', side_effect=simple_mock) as mock:
        yield mock


class TestAuthentication:
    
    def test_missing_authorization_header(self, mock_generate_events):
        """Test request without Authorization header"""
        response = client.post(
            "/pipeline/", 
            json={"names": ["Betnovate Scalp Application", "Panadol"]}
        )
        
        assert response.status_code in [401, 403]  
    
    def test_invalid_authorization_format(self, mock_generate_events):
        """Test malformed Authorization header"""
        headers = {"Authorization": "InvalidFormat token123"}
        response = client.post(
            "/pipeline/", 
            headers=headers
        )
        
        assert response.status_code == 403

    @patch.dict(os.environ, {"AUTH_API_KEY": "test_key_123"}) 
    def test_invalid_api_key(self, mock_generate_events):
        """Test with wrong API key"""
        headers = {"Authorization": "Bearer wrong_key"}
        response = client.post(
            "/pipeline/", 
            headers=headers
        )
        
        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]
    
    @patch.dict(os.environ, {"AUTH_API_KEY": "test_key_123"})
    def test_valid_api_key(self, mock_generate_events):
        """Test with correct API key"""

        headers = {"Authorization": "Bearer test_key_123"}
        response = client.post(
            "/pipeline/", 
            json={"names": ["test"]},
            headers=headers
        )
        assert response.status_code == 200
