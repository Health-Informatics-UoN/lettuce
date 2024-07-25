import logging

import torch


def get_model(
    model_name: str, temperature: float = 0.7, logger: logging.Logger | None = None
) -> object:
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
    # TODO: Llama dictionary of args?
    # TODO: error handling if the model isn't one of those listed
    if "gpt" in model_name.lower():
        logger.info(f"Loading {model_name} model")
        from haystack.components.generators import OpenAIGenerator

        llm = OpenAIGenerator(
            model=model_name, generation_kwargs={"temperature": temperature}
        )

    elif "llama" in model_name.lower() or "gemma" in model_name.lower():
        if model_name.lower() == "llama-2-7b-chat":
            """
            [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
            [GGUF format](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)
            """
            repo = "TheBloke/Llama-2-7B-Chat-GGUF"
            filename = "llama-2-7b-chat.Q4_0.gguf"  # Options: llama-2-7b-chat.Q4_0.gguf, llama-2-7b-chat.Q5_0.gguf, llama-2-7b-chat.Q8_0.gguf

        elif model_name.lower() == "llama-3-8b":
            """
            [Llama-3](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
            [GGUF format](https://huggingface.co/QuantFactory/Meta-Llama-3-8B-GGUF-v2)
            """
            repo = "QuantFactory/Meta-Llama-3-8B-GGUF-v2"
            filename = "Meta-Llama-3-8B.Q4_K_M.gguf"

        elif model_name.lower() == "llama-3-70b":
            """
            [Llama-3](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)
            [GGUF format](https://huggingface.co/QuantFactory/Meta-Llama-3-70B-Instruct-GGUF-v2)
            """
            repo = "QuantFactory/Meta-Llama-3-70B-Instruct-GGUF-v2"
            filename = "Meta-Llama-3-70B-Instruct-v2.Q4_K_M.gguf"

        elif model_name.lower() == "gemma-7b":
            """
            [Gemma-7b](https://huggingface.co/google/gemma-7b)
            [GGUF format](https://huggingface.co/MaziyarPanahi/gemma-7b-GGUF)
            """
            repo = "MaziyarPanahi/gemma-7b-GGUF"
            filename = "gemma-7b.Q4_K_M.gguf"

        logger.info(f"Loading {model_name} model")
        from haystack_integrations.components.generators.llama_cpp import (
            LlamaCppGenerator,
        )
        from huggingface_hub import hf_hub_download

        device = -1 if torch.cuda.is_available() else 0
        llm = LlamaCppGenerator(
            model=hf_hub_download(
                repo_id=repo,
                filename=filename,
            ),
            n_ctx=0,  # Text context, 0 = from model
            n_batch=512,
            model_kwargs={"n_gpu_layers": device, "verbose": True},
            generation_kwargs={"max_tokens": 128, "temperature": temperature},
        )

    return llm
