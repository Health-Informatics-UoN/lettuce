import logging
from haystack.components.generators import OpenAIGenerator
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
import torch


# ---------- LOCAL MODELS DICTIONARY ---------- >
# --------------------------------------------------- >

local_models = {
    # ------ llama-2-7b-chat ------- >
    "llama-2-7b-chat": {
        "repo_id": "TheBloke/Llama-2-7B-Chat-GGUF",
        "filename": "llama-2-7b-chat.Q4_0.gguf",
    },
    # ------ llama-3-8b ------- >
    "llama-3-8b": {
        "repo_id": "QuantFactory/Meta-Llama-3-8B-GGUF-v2",
        "filename": "Meta-Llama-3-8B.Q4_K_M.gguf",
    },
    # ------ llama-3-70b ------- >
    "llama-3-70b": {
        "repo_id": "QuantFactory/Meta-Llama-3-70B-Instruct-GGUF-v2",
        "filename": "Meta-Llama-3-70B-Instruct-v2.Q4_K_M.gguf",
    },
    # ------ gemma-7b ------- >
    "gemma-7b": {
        "repo_id": "MaziyarPanahi/gemma-7b-GGUF",
        "filename": "gemma-7b.Q4_K_M.gguf",
    },
    # ------ llama-3.1-8b ------- >
    "llama-3.1-8b": {
        "repo_id": "MaziyarPanahi/Meta-Llama-3.1-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf",
    },
    # ------ mistral-7b ------- >
    "mistral-7b": {
        "repo_id": "TheBloke/Mistral-7B-GGUF",
        "filename": "mistral-7b.Q4_K_M.gguf",
    },
    # ------ kuchiki-l2-7b ------- >
    "kuchiki-l2-7b": {
        "repo_id": "TheBloke/Kuchiki-L2-7B-GGUF",
        "filename": "kuchiki-l2-7b.Q4_K_M.gguf",
    },
    # ------ tinyllama-1.1b-chat ------- >
    "tinyllama-1.1b-chat": {
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF",
        "filename": "tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf",
    },
    # ------ bioMistral-7B-GGUF ------- >
    "biomistral-7b": {
        "repo_id": "MaziyarPanahi/BioMistral-7B-GGUF",
        "filename": "BioMistral-7B.Q4_K_M.gguf",
    },
    # ------ Qwen2.5-3B-Instruct-GGUF ------- >
    "qwen2.5-3b-instruct": {
        "repo_id": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "filename": "qwen2.5-3b-instruct-q5_k_m.gguf",
    },
    # ------ airoboros-3b-3p0-GGUF ------- >
    "airoboros-3b": {
        "repo_id": "afrideva/airoboros-3b-3p0-GGUF",
        "filename": "airoboros-3b-3p0.q4_k_m.gguf",
    },
    # ------ medicine-chat ------- >
    "medicine-chat": {
        "repo_id": "TheBloke/medicine-chat-GGUF",
        "filename": "medicine-chat.Q4_K_M.gguf",
    },
    "medicine-llm-13b": {
        "repo_id": "TheBloke/medicine-LLM-13B-GGUF",
        "filename": "medicine-llm-13b.Q3_K_S.gguf",
    },
    # ------ med llama Q5_K_S (5.59GB) ------- >
    "med-llama-3-8b-v1": {
        "repo_id": "bartowski/JSL-MedLlama-3-8B-v1.0-GGUF",
        "filename": "JSL-MedLlama-3-8B-v1.0-Q5_K_S.gguf",
    },
    # ------ med llama Q4_K_M (4.92GB) ------- >
    "med-llama-3-8b-v2": {
        "repo_id": "bartowski/JSL-MedLlama-3-8B-v1.0-GGUF",
        "filename": "JSL-MedLlama-3-8B-v1.0-Q4_K_M.gguf",
    },
    # ------ med llama Q3_K_M (4.01GB) ------- >
    "med-llama-3-8b-v3": {
        "repo_id": "bartowski/JSL-MedLlama-3-8B-v1.0-GGUF",
        "filename": "JSL-MedLlama-3-8B-v1.0-Q3_K_M.gguf",
    },
    # ------ med llama IQ3_M (3.78GB) ------- >
    "med-llama-3-8b-v4": {
        "repo_id": "bartowski/JSL-MedLlama-3-8B-v1.0-GGUF",
        "filename": "JSL-MedLlama-3-8B-v1.0-IQ3_M.gguf",
    },
    # ------ Add more models here ------ >
}


# ---------- OTHER MODELS THAT CAN BE CONSIDERED ---------- >

"""

GPT-NeoX: 

    Model Type: Transformer Based LLM by EleutherAI
    
    Parameters: 
        {
            GPT-NeoX-6B : 6 billion (6B),
            GPT-Neo-20B : 20 billion (20B)
        }

"""


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
