import logging
from haystack.components.generators import OpenAIGenerator
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
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
        "filename": "Llama-3.2-3B-Instruct-Q6_K_L.gguf",
    },
}


def get_model(
    model_name: str, temperature: float = 0.7, logger: logging.Logger | None = None
) -> OpenAIGenerator | LlamaCppGenerator:
    """
    Get an interface for interacting with an LLM

    Uses Haystack Generators to provide an interface to a model.
    If the model_name is a GPT, then the interface is to a remote OpenAI model. Otherwise, uses a LlamaCppGenerator to start a llama.cpp model and provide an interface.

    Parameters
    ----------
    model_name: str
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
        except:
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
