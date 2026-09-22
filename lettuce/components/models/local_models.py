import os 
import logging
from huggingface_hub import hf_hub_download
from options.base_options import BaseOptions

settings = BaseOptions()

try:
    from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
    import torch
except ImportError:
    raise ImportError("To use a Llama.cpp generator you have to install one of the optional dependency groups. Consult the documentation for details.")

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
    filename: str,
    repo_id: str,
    temperature: float, 
    logger: logging.Logger, 
    verbose: bool,
    n_ctx: int = 1024,
    n_batch: int = 32,
    max_tokens: int = 128 
): 
    """
    Load GGUF model weights from a hugging face repository.

    Parameters
    ----------
    filename: str
        The filename within a huggingface hub repository to download
    repo_id: str
        The ID of a huggingface hub repository to fetch the filename from
    temperature: float
        The temperature for model generation
    logger : logging.Logger
        Logger instance for tracking progress and errors.
    verbose: bool
        If true, the generator logs information about loading weights and generation
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
    logger.info(f"Loading local model: {filename}")
    device = -1 if (torch.cuda.is_available() or torch.backends.mps.is_available()) else 0
    logger.info(f"Using {device} GPU layers")

    try: 
        model_path = hf_hub_download(repo_id=repo_id, filename=filename) 
    except Exception as e: 
        logger.error(f"Failed to download model {filename}: {str(e)}")
        raise ValueError(f"Failed to load model {filename}: {str(e)}")
    
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
        logger.error(f"Failed to initialize LlamaCppGenerator for {filename}: {str(e)}")
        raise ValueError(f"Failed to initialize local model {filename}: {str(e)}")

    return llm 

