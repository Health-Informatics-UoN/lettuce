import os 
import logging
from haystack.components.generators import OpenAIGenerator
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
from huggingface_hub import hf_hub_download
from options.pipeline_options import LLMModel
import torch


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

FALLBACK_MODEL = "llama-3.1-8b"

def get_local_weights(
    path_to_weights: str, 
    temperature: float, 
    logger: logging.Logger
):
    """
    Load a local GGUF model weights file and return a LlamaCppGenerator object.

    Parameters
    ----------
    path_to_weights : str
        The full path to the local GGUF model weights file (e.g., "/path/to/llama-2-7b-chat.Q4_0.gguf").
    temperature : float, optional
        The temperature for model generation (default is 0.7).
    logger : logging.Logger
        Logger instance for tracking progress and errors.

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
    device = -1 if torch.cuda.is_available() else 0 

    # Load the model using llama 
    llm = LlamaCppGenerator(
        model=path_to_weights, 
        n_ctx=0, 
        n_batch=512, 
        model_kwargs={"n_gpu_layers": device, "verbose": True}, 
        generation_kwargs={"max_tokens": 128, "temperature": temperature}
    )
    logger.info(f"Succesfully loaded LlamaCppGenerator from {path_to_weights}")
    return llm 


def download_model_from_huggingface(
    model_name: str, 
    temperature: float, 
    logger: logging.Logger 
): 
    logger.info(f"Loading local model: {model_name}")
    device = -1 if torch.cuda.is_available() else 0

    try: 
        model_config = local_models[model_name]
        model_path = hf_hub_download(**model_config) 
    except KeyError: 
        logger.warning(f"Model {model_name} not found in local_models. Falling back to {FALLBACK_MODEL}")
        model_config = local_models[FALLBACK_MODEL]
        model_path = hf_hub_download(**model_config)
    except Exception as e: 
        logger.error(f"Failed to download model {model_name}: {str(e)}")
        raise ValueError(f"Failed to load local model {model_name}: {str(e)}")
    
    try: 
        llm = LlamaCppGenerator(
            model=model_path, 
            n_ctx=0, 
            n_batch=512, 
            model_kwargs={"n_gpu_layers": device, "verbose": True}, 
            generation_kwargs={"max_tokens": 128, "temperature": temperature}
        )
    except Exception as e: 
        logger.error(f"Failed to initialize LlamaCppGenerator for {model_name}: {str(e)}")
        raise ValueError(f"Failed to initialize local model {model_name}: {str(e)}")

    return llm 


def get_model(
    model: LLMModel, 
    logger: logging.Logger, 
    temperature: float = 0.7
) -> OpenAIGenerator | LlamaCppGenerator:
    """
    Get an interface for interacting with an LLM

    Uses Haystack Generators to provide an interface to a model.
    If the model_name is a GPT, then the interface is to a remote OpenAI model. Otherwise, uses a LlamaCppGenerator to start a llama.cpp model and provide an interface.

    Parameters
    ----------
    model: LLMModel
        The name of the model
    temperature: float
        The temperature for the model
    logger: logging.Logger|None
        The logger for the model

    Returns
    -------
    object
        An interface to generate text using an LLM
    """
    path_to_local_weights = os.getenv("LOCAL_LLM")
    model_name = model.value
    if path_to_local_weights: 
        llm = get_local_weights(path_to_local_weights, temperature, logger)
    else: 
        if "gpt" in model_name.lower():
            logger.info(f"Loading {model_name} model")

            llm = OpenAIGenerator(
                model=model_name, generation_kwargs={"temperature": temperature}
            )

        else:
            logger.info(f"Loading {model_name} model")
            from huggingface_hub import hf_hub_download

            device = -1 if torch.cuda.is_available() else 0

            try:
                llm = LlamaCppGenerator(
                    model=hf_hub_download(**local_models[model_name]),
                    n_ctx=0,  # Text context, 0 = from model
                    n_batch=512,
                    model_kwargs={"n_gpu_layers": device, "verbose": True},
                    generation_kwargs={"max_tokens": 128, "temperature": temperature},
                )
            except KeyError:
                print(f"{model_name} is not a recognised model name")
                print(f"Error loading {model_name}")
            finally:
                logger.info("Loading llama-3.1-8b")
                llm = LlamaCppGenerator(
                    model=hf_hub_download(**local_models[model_name]),
                    n_ctx=0,
                    n_batch=512,
                    model_kwargs={"n_gpu_layers": device, "verbose": True},
                    generation_kwargs={"max_tokens": 128, "temperature": temperature},
                )

    return llm
