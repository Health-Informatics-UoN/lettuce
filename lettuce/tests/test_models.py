import pytest 
from unittest.mock import patch, Mock, MagicMock 
import logging 
import os 
from components.models import (
    get_local_weights, 
    download_model_from_huggingface, 
    download_model_from_openai, 
    get_model, 
    local_models
)
from options.pipeline_options import LLMModel 

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def mock_llm_model(): 
    """Fixture to provide a mock LLMModel object."""
    mock_model = MagicMock(spec=LLMModel) 
    mock_model.value = "llama-2-7b-chat"
    return mock_model


@pytest.fixture 
def mock_file_exists(tmp_path): 
    """Fixture to create a temporary file path for testing."""
    file_path = tmp_path / "model.gguf"
    file_path.touch() 
    return str(file_path)
 

@patch("components.get_model.os.path.isfile")
@patch("components.get_model.LlamaCppGenerator")
@patch("components.get_model.torch.cuda.is_available")
def test_local_weights_success(mock_cuda, mock_llama, mock_isfile, mock_file_exists):
    mock_isfile.return_value = True 
    mock_llm_instance = Mock() 
    mock_llama.return_value = mock_llm_instance 
    temperature = 0.7 

    result = get_local_weights(mock_file_exists, temperature, logger)

    assert result == mock_llm_instance 
    mock_isfile.assert_called_once_with(mock_file_exists)
    mock_llama.assert_called_once_with(
        model=mock_file_exists, 
        n_ctx=0, 
        n_batch=512, 
        model_kwargs={"n_gpu_layers": -1 if mock_cuda.return_value else 0, "verbose": True}, 
        generation_kwargs={"max_tokens": 128, "temperature": temperature}
    )
    mock_cuda.assert_called_once()


def test_get_local_weights_file_not_found(): 
    pass 


def test_download_model_from_huggingface_success(): 
    pass 


def test_download_model_from_huggingface_fallback(): 
    pass 


def test_download_model_from_huggingface_download_error(): 
    pass 


def test_download_model_from_openai_success(): 
    pass 
