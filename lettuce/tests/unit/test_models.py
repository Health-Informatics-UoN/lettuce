import pytest 
from unittest.mock import patch, Mock, MagicMock 
import logging 
from components.models import (
    get_local_weights, 
    download_model_from_huggingface, 
    connect_to_openai, 
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
 

@patch("components.models.os.path.isfile")
@patch("components.models.LlamaCppGenerator")
@patch("components.models.torch.cuda.is_available")
def test_local_weights_success(mock_cuda, mock_llama, mock_isfile, mock_file_exists):
    mock_isfile.return_value = True 
    mock_llm_instance = Mock() 
    mock_llama.return_value = mock_llm_instance 
    temperature = 0.7 

    result = get_local_weights(mock_file_exists, temperature, logger, verbose=False)

    assert result == mock_llm_instance 
    mock_isfile.assert_called_once_with(mock_file_exists)
    mock_llama.assert_called_once_with(
        model=mock_file_exists, 
        model_kwargs={'n_ctx': 1024, 'n_batch': 32, 'n_gpu_layers': -1, 'verbose': False},
        generation_kwargs={'max_tokens': 128, 'temperature': 0.7}
    )
    mock_cuda.assert_called_once()


@patch("components.models.os.path.isfile")
def test_get_local_weights_file_not_found(mock_isfile): 
    mock_isfile.return_value = False 
    path = "/fake/path/model.gguf"
    temperature = 0.7 

    with pytest.raises(FileNotFoundError) as exc_info: 
        get_local_weights(path, temperature, logger, verbose=False)
    assert str(exc_info.value) == f"Model weights file not found at {path}"
    mock_isfile.assert_called_once_with(path) 


@patch("components.models.hf_hub_download")
@patch("components.models.LlamaCppGenerator")
def test_download_model_from_huggingface_success(mock_llama, mock_download): 
    mock_download.return_value = "/downloaded/fallback.gguf"
    mock_llm_instance = Mock()
    mock_llama.return_value = mock_llm_instance
    model_name = "unknown-model"
    temperature = 0.7
    fallback_model = "llama-3.1-8b"

    result = download_model_from_huggingface(model_name, temperature, logger, fallback_model=fallback_model, verbose=False)

    assert result == mock_llm_instance
    mock_download.assert_called_once_with(**local_models[fallback_model])
    mock_llama.assert_called_once()


@patch("components.models.hf_hub_download")
@patch("components.models.LlamaCppGenerator")
def test_download_model_from_huggingface_fallback(mock_llama, mock_download):
    mock_download.return_value = "/downloaded/fallback.gguf"
    mock_llm_instance = Mock()
    mock_llama.return_value = mock_llm_instance
    model_name = "unknown-model"
    temperature = 0.7
    fallback_model = "llama-3.1-8b"

    result = download_model_from_huggingface(model_name, temperature, logger, fallback_model=fallback_model, verbose=False)

    assert result == mock_llm_instance
    mock_download.assert_called_once_with(**local_models[fallback_model])
    mock_llama.assert_called_once()


@patch("components.models.hf_hub_download")
def test_download_model_from_huggingface_download_error(mock_download):
    mock_download.side_effect = Exception("Download error")
    model_name = "llama-2-7b-chat"
    temperature = 0.7

    with pytest.raises(ValueError) as exc_info:
        download_model_from_huggingface(model_name, temperature, logger, verbose=False)
    assert "Failed to load model" in str(exc_info.value)
    mock_download.assert_called_once()


@patch("components.models.OpenAIGenerator")
def test_connect_to_openai_success(mock_openai): 
    mock_llm_instance = Mock()
    mock_openai.return_value = mock_llm_instance 
    model_name = "gpt-3.5-turbo"
    temperature = 0.7 

    result = connect_to_openai(model_name, temperature, logger)

    assert result == mock_llm_instance
    mock_openai.assert_called_once_with(
        model=model_name,
        generation_kwargs={"temperature": temperature}
    )


@patch("components.models.get_local_weights")
def test_get_model_local_weights(mock_get_local, mock_llm_model): 
    fake_path_to_local_weights = "/fake/path/model.gguf"
    mock_llm_instance = Mock()
    mock_get_local.return_value = mock_llm_instance

    result = get_model(mock_llm_model, logger, path_to_local_weights=fake_path_to_local_weights)

    assert result == mock_llm_instance
    mock_get_local.assert_called_once_with(fake_path_to_local_weights, 0.7, logger, False)


@patch("components.models.connect_to_openai")
def test_get_model_openai(mock_openai_download, mock_llm_model): 
    mock_llm_instance = Mock()
    mock_openai_download.return_value = mock_llm_instance
    mock_llm_model.value = "gpt-3.5-turbo"

    result = get_model(mock_llm_model, logger)

    assert result == mock_llm_instance
    mock_openai_download.assert_called_once_with("gpt-3.5-turbo", 0.7, logger)


@patch("components.models.download_model_from_huggingface")
def test_get_model_huggingface(mock_hf_download, mock_llm_model):
    mock_llm_instance = Mock()
    mock_hf_download.return_value = mock_llm_instance

    result = get_model(mock_llm_model, logger)

    assert result == mock_llm_instance
    mock_hf_download.assert_called_once_with("llama-2-7b-chat", 0.7, logger, False)
