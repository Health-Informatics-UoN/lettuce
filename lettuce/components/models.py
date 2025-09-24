import os 
import logging
from typing import Any 
from haystack.components.generators import OpenAIGenerator
from haystack_integrations.components.generators.ollama import OllamaGenerator
from huggingface_hub import hf_hub_download
from options.base_options import BaseOptions
from options.pipeline_options import InferenceType, LLMModel
import torch

settings = BaseOptions()

if settings.inference_type == InferenceType.LLAMA_CPP:
    try:
        from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
    except ImportError:
        raise ImportError("To use a Llama.cpp generator you have to install one of the optional dependency groups. Consult the documentation for details.")

local_models = {
    "llama-2-7b-chat": {
        "repo_id": "TheBloke/Llama-2-7B-Chat-GGUF",
        "filename": "llama-2-7b-chat.Q4_0.gguf",
    },
    "llama-3-8b": {
        "repo_id": "QuantFactory/Meta-Llama-3-8B-GGUF-v2",
        "filename": "Meta-Llama-3-8B.Q4_K_M.gguf",
    },
    "llama-3-70b": {
        "repo_id": "QuantFactory/Meta-Llama-3-70B-Instruct-GGUF-v2",
        "filename": "Meta-Llama-3-70B-Instruct-v2.Q4_K_M.gguf",
    },
    "gemma-7b": {
        "repo_id": "MaziyarPanahi/gemma-7b-GGUF",
        "filename": "gemma-7b.Q4_K_M.gguf",
    },
    "llama-3.1-8b": {
        "repo_id": "MaziyarPanahi/Meta-Llama-3.1-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf",
    },
    "llama-3.2-3b": {
        "repo_id": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "filename": "Llama-3.2-3B-Instruct-Q6_K.gguf",
    },
    "mistral-7b": {
        "repo_id": "TheBloke/Mistral-7B-GGUF",
        "filename": "mistral-7b.Q4_K_M.gguf",
    },
    "kuchiki-l2-7b": {
        "repo_id": "TheBloke/Kuchiki-L2-7B-GGUF",
        "filename": "kuchiki-l2-7b.Q4_K_M.gguf",
    },
    "tinyllama-1.1b-chat": {
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF",
        "filename": "tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf",
    },
    "biomistral-7b": {
        "repo_id": "MaziyarPanahi/BioMistral-7B-GGUF",
        "filename": "BioMistral-7B.Q4_K_M.gguf",
    },
    "qwen2.5-3b-instruct": {
        "repo_id": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "filename": "qwen2.5-3b-instruct-q5_k_m.gguf",
    },
    "airoboros-3b": {
        "repo_id": "afrideva/airoboros-3b-3p0-GGUF",
        "filename": "airoboros-3b-3p0.q4_k_m.gguf",
    },
    "medicine-chat": {
        "repo_id": "TheBloke/medicine-chat-GGUF",
        "filename": "medicine-chat.Q4_K_M.gguf",
    },
    "medicine-llm-13b": {
        "repo_id": "TheBloke/medicine-LLM-13B-GGUF",
        "filename": "medicine-llm-13b.Q3_K_S.gguf",
    },
    "med-llama-3-8b-v1": {
        "repo_id": "bartowski/JSL-MedLlama-3-8B-v1.0-GGUF",
        "filename": "JSL-MedLlama-3-8B-v1.0-Q5_K_S.gguf",
    },
    "med-llama-3-8b-v2": {
        "repo_id": "bartowski/JSL-MedLlama-3-8B-v1.0-GGUF",
        "filename": "JSL-MedLlama-3-8B-v1.0-Q4_K_M.gguf",
    },
    "med-llama-3-8b-v3": {
        "repo_id": "bartowski/JSL-MedLlama-3-8B-v1.0-GGUF",
        "filename": "JSL-MedLlama-3-8B-v1.0-Q3_K_M.gguf",
    },
    "med-llama-3-8b-v4": {
        "repo_id": "bartowski/JSL-MedLlama-3-8B-v1.0-GGUF",
        "filename": "JSL-MedLlama-3-8B-v1.0-IQ3_M.gguf",
    },
}


def get_local_weights(
    path_to_weights: os.PathLike | str | None, 
    temperature: float, 
    logger: logging.Logger,
    verbose: bool
):
    """
    Load a local GGUF model weights file and return a LlamaCppGenerator object.

    Parameters
    ----------
    path_to_weights : os.PathLike | str | None
        The full path to the local GGUF model weights file (e.g., "/path/to/llama-2-7b-chat.Q4_0.gguf").
    temperature : float
        The temperature for model generation
    logger : logging.Logger
        Logger instance for tracking progress and errors.
    verbose: bool
        If true, the generator logs information about loading weights and generation

    Returns
    -------
    LlamaCppGenerator
        A loaded LlamaCppGenerator object ready for inference.

    Raises
    ------
    FileNotFoundError
        If the specified file_path does not exist or is not a file.
    """
    if not os.path.isfile(path_to_weights):
        logger.error(f"Model weights not found at {path_to_weights}")
        raise FileNotFoundError(f"Model weights file not found at {path_to_weights}")
   
    logger.info(f"Loading local model weights from {path_to_weights}")
    device = -1 if (torch.cuda.is_available() or torch.backends.mps.is_available()) else 0
    logger.info(f"Using {device} GPU layers")

    # Load the model using llama 
    llm = LlamaCppGenerator(
        model=path_to_weights, 
        model_kwargs={
            "n_ctx": 1024,
            "n_batch": 32,
            "n_gpu_layers": device,
            "verbose": verbose
        }, 
        generation_kwargs={"max_tokens": 128, "temperature": temperature}
    )
    logger.info(f"Succesfully loaded LlamaCppGenerator from {path_to_weights}")
    return llm 


def download_model_from_huggingface(
    model_name: str, 
    temperature: float, 
    logger: logging.Logger, 
    verbose: bool,
    fallback_model: str = "llama-3.1-8b",
    n_ctx: int = 1024,
    n_batch: int = 32,
    max_tokens: int = 128 
): 
    """
    Load GGUF model weights from a hugging face repository.

    Parameters
    ----------
    model_name: str
        The name of a model with repository details in the local_models dictionary
    temperature: float
        The temperature for model generation
    logger : logging.Logger
        Logger instance for tracking progress and errors.
    verbose: bool
        If true, the generator logs information about loading weights and generation
    fallback_model: str
        If the model name that's specified is not in the local_models dictionary, loads this one instead. Defaults to llama-3.1-8b
    n_ctx: int
        Context size for the model
    n_batch: int
        Number of tokens sent to the model in each batch. Defaults to 32
    max_tokens: int
        Maximum tokens to generate. Defaults to 128.

    Returns
    -------
    LlamaCppGenerator
        A loaded LlamaCppGenerator object ready for inference.

    Raises
    ------
    ValueError
        If the model fails to download or initialize.
    """
    logger.info(f"Loading local model: {model_name}")
    device = -1 if (torch.cuda.is_available() or torch.backends.mps.is_available()) else 0
    logger.info(f"Using {device} GPU layers")

    try: 
        model_config = local_models[model_name]
        model_path = hf_hub_download(**model_config) 
    except KeyError: 
        logger.warning(f"Model {model_name} not found in local_models. Falling back to {fallback_model}")
        model_config = local_models[fallback_model]
        model_path = hf_hub_download(**model_config)
    except Exception as e: 
        logger.error(f"Failed to download model {model_name}: {str(e)}")
        raise ValueError(f"Failed to load model {model_name}: {str(e)}")
    
    try: 
        llm = LlamaCppGenerator(
            model=model_path, 
            model_kwargs={
                "n_ctx": n_ctx,
                "n_batch": n_batch,
                "n_gpu_layers": device,
                "verbose": verbose,
            },
            generation_kwargs={"max_tokens": max_tokens, "temperature": temperature}
        )
    except Exception as e: 
        logger.error(f"Failed to initialize LlamaCppGenerator for {model_name}: {str(e)}")
        raise ValueError(f"Failed to initialize local model {model_name}: {str(e)}")

    return llm 


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
    url: str,
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
    model_name = model.value
    # I know a match might seem like overkill, this is in case other inference engines are added
    match inference_type:
        case InferenceType.OPEN_AI:
            llm = connect_to_openai(model_name, temperature, logger)
        case InferenceType.OLLAMA:
            llm = connect_to_ollama(model_name, url, temperature, logger)
        case _:
            if path_to_local_weights:
                llm = get_local_weights(path_to_local_weights, temperature, logger, verbose)
            else:
                llm = download_model_from_huggingface(model_name, temperature, logger, verbose)
    return llm
