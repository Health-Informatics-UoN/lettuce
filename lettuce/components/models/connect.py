import os 
import logging
from typing import Any 
from haystack.components.generators import OpenAIGenerator
from haystack_integrations.components.generators.ollama import OllamaGenerator
from options.pipeline_options import InferenceType, LLMModel

def connect_to_openai(
    model_name: str, 
    temperature: float, 
    logger: logging.Logger,
):
    """
    Connect to OpenAI API and return an OpenAIGenerator object.

    Parameters
    ----------
    model_name : str
        The name of the OpenAI model (e.g., "gpt-4", "gpt-3.5-turbo")
    temperature : float
        The temperature for model generation
    logger : logging.Logger
        Logger instance for tracking progress and errors.

    Returns
    -------
    OpenAIGenerator
        A configured OpenAIGenerator object for API-based inference.
    """
    logger.info(f"Loading {model_name} model")
    llm = OpenAIGenerator(
        model=model_name, generation_kwargs={"temperature": temperature}
    )
    return llm 


def connect_to_ollama(
    model_name: str,
    url: str,
    temperature: float,
    logger: logging.Logger,
    max_tokens: int = 128,
):
    """
    Connect to an Ollama server and return an OllamaGenerator object.

    Parameters
    ----------
    model_name : str
        The name of the Ollama model to use
    url : str
        The URL of the Ollama server
    temperature : float
        The temperature for model generation
    logger : logging.Logger
        Logger instance for tracking progress and errors.
    max_tokens : int
        Maximum number of tokens to generate. Defaults to 128.

    Returns
    -------
    OllamaGenerator
        A configured OllamaGenerator object for Ollama server-based inference.

    Raises
    ------
    Exception
        If connection to the Ollama server fails or the model is not available.
    """
    logger.info(f"Loading Ollama model: {model_name}")
    try:
        return OllamaGenerator(
            model=model_name,
            url=url,
            generation_kwargs = {
                "max_tokens": max_tokens,
                "temperature": temperature
                }
            )
    except Exception as e:
        logger.error(f"Couldn't communicate with an Ollama server: {str(e)} Is it running? Have you pulled {model_name} before?")
        raise


def get_model(
    model: LLMModel, 
    logger: logging.Logger, 
    inference_type: InferenceType,
    url: str|None,
    temperature: float = 0.7, 
    path_to_local_weights: os.PathLike[Any] | str | None = None,
    verbose: bool = False,
):
    """
    Get an interface for interacting with an LLM.

    Uses Haystack Generators to provide an interface to a model.

    Parameters
    ----------
    model: LLMModel
        The name of the model
    logger: logging.Logger
        The logger for the model
    inference_type: InferenceType
        Whether to use Llama.cpp, Ollama, or the OpenAI API for inference
    url: str
        The URL for the Ollama server (only used when inference_type is OLLAMA)
    temperature: float
        The temperature for the model. Defaults to 0.7
    path_to_local_weights: os.PathLike | str | None
        Filepath to load weights locally. If not provided will default to downloading model weights.
    verbose: bool
        If true, the generator logs information about loading weights and generation. Defaults to False

    Returns
    -------
    OpenAIGenerator | LlamaCppGenerator | OllamaGenerator
        An interface to generate text using an LLM
    """
    # I know a match might seem like overkill, this is in case other inference engines are added
    match inference_type:
        case InferenceType.OPEN_AI:
            llm = connect_to_openai(model.value, temperature, logger)
        case InferenceType.OLLAMA:
            llm = connect_to_ollama(model.ollama_spec, url, temperature, logger)
        case InferenceType.LLAMA_CPP:
            from .local_models import get_local_weights, download_model_from_huggingface
            if path_to_local_weights:
                llm = get_local_weights(path_to_local_weights, temperature, logger, verbose)
            else:
                llm = download_model_from_huggingface(model.filename, model.repo_id, temperature, logger, verbose)
    return llm
