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
    # ------ pythia-70m ------- >
    "pythia-70m": {
        "repo_id": "EleutherAI/pythia-70m-GGUF",
        "filename": "pythia-70m.Q4_K_M.gguf",
    },
    # ------ pythia-410m ------- >
    "pythia-410m": {
        "repo_id": "EleutherAI/pythia-410m-GGUF",
        "filename": "pythia-410m.Q4_K_M.gguf",
    },
    # ------ pythia-1b ------- >
    "pythia-1b": {
        "repo_id": "EleutherAI/pythia-1b-GGUF",
        "filename": "pythia-1b.Q4_K_M.gguf",
    },
    # ------ pythia-1.4b ------- >
    "pythia-1.4b": {
        "repo_id": "EleutherAI/pythia-1.4b-GGUF",
        "filename": "pythia-1.4b.Q4_K_M.gguf",
    },
    # ------ pythia-2.8b ------- >
    "pythia-2.8b": {
        "repo_id": "EleutherAI/pythia-2.8b-GGUF",
        "filename": "pythia-2.8b.Q4_K_M.gguf",
    },
    # ------ alpaca-lora-7b ------- >
    "alpaca-lora-7b": {
        "repo_id": "TheBloke/Alpaca-LoRA-7B-GGUF",
        "filename": "alpaca-lora-7b.Q4_K_M.gguf",
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

    If the model_name is a GPT, then the interface is to a remote
    OpenAI model. Otherwise, uses a LlamaCppGenerator to start a
    llama.cpp model and provide an interface.

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
    # Normalize the huggingface for downloading the model
    from huggingface_hub import hf_hub_download

    # Normalize the model name to lowercase
    model_name = model_name.lower()

    # Select device based on GPU availability
    device = -1 if torch.cuda.is_available() else 0

    try:
        # ------ GPT Model ------ >
        # --------------------------

        if "gpt" in model_name:
            logger.info(f"Loading {model_name} from OpenAI API")
            try:
                llm = OpenAIGenerator(
                    model=model_name, generation_kwargs={"temperature": temperature}
                )
            except Exception as e:
                logger.error(f"Failed to load GPT-based model {model_name}: {e}")
                raise ValueError(f"Failed to load GPT-based model {model_name}: {e}")

        # ------ Llama Model (7B) ------ >
        # --------------------------

        elif model_name == "llama2-7b":
            """
            llama-2-7b-chat
            ----------------

                - Model Type: LLaMA (Large Language Model Meta AI)

                - Parameters: 7 billion (7B)

                - Description:
                    Llama-2 is an open-access language model developed by
                    Meta AI, optimized for both chat and general text generation tasks.
            """

            logger.info(f"Loading {model_name} from local models")
            try:
                llm = LlamaCppGenerator(
                    model=hf_hub_download(**local_models["llama-2-7b-chat"]),
                    n_ctx=0,
                    n_batch=512,
                    model_kwargs={"n_gpu_layers": device, "verbose": True},
                    generation_kwargs={"max_tokens": 128, "temperature": temperature},
                )
            except Exception as e:
                logger.error(f"Failed to load llama2-7b: {e}")
                raise ValueError(f"Failed to load llama2-7b: {e}")

        # ------ Llama Model (8B) ------ >
        # --------------------------

        elif model_name == "llama3-8b":
            """
            llama-3-8b
            -----------

                - Model Type: LLaMA (Large Language Model Meta AI)

                - Parameters: 8 billion (8B)

                - Description:
                    A slightly larger model than Llama-2-7B, offering improved
                    instruction-following capabilities while still being manageable
                    in size.
            """

            logger.info("Loading llama3-8b model locally")
            try:
                llm = LlamaCppGenerator(
                    model=hf_hub_download(**local_models["llama-3-8b"]),
                    n_ctx=0,
                    n_batch=512,
                    model_kwargs={"n_gpu_layers": device, "verbose": True},
                    generation_kwargs={"max_tokens": 128, "temperature": temperature},
                )
            except Exception as e:
                logger.error(f"Failed to load llama3-8b: {e}")
                raise ValueError(f"Failed to load llama3-8b: {e}")

        # ------ Llama Model (70B) ------ >
        # --------------------------

        elif model_name == "llama3-70b":
            """
            llama-3-70b
            ------------

                - Model Type: LLaMA (Large Language Model Meta AI)

                - Parameters: 70 billion (70B)

                - Description:
                    Offers superior performance in complex tasks such as deep text
                    understanding and sophisticated natural language generation.
            """

            logger.info("Loading llama3-70b model locally")
            try:
                llm = LlamaCppGenerator(
                    model=hf_hub_download(**local_models["llama-3-70b"]),
                    n_ctx=0,
                    n_batch=512,
                    model_kwargs={"n_gpu_layers": device, "verbose": True},
                    generation_kwargs={"max_tokens": 128, "temperature": temperature},
                )
            except Exception as e:
                logger.error(f"Failed to load llama3-70b: {e}")
                raise ValueError(f"Failed to load llama3-70b: {e}")

        # ------ Gemma Model (7B) -----
        # --------------------------

        elif model_name == "gemma-7b":
            """
            gemma-7b
            ---------

                - Model Type: LLaMA (Large Language Model Meta AI)

                - Parameters: 7 billion (7B)

                - Description:
                    Compact model with good general-purpose performance,
                    especially in conversational tasks.
            """

            logger.info("Loading gemma-7b model locally")
            try:
                llm = LlamaCppGenerator(
                    model=hf_hub_download(**local_models["gemma-7b"]),
                    n_ctx=0,
                    n_batch=512,
                    model_kwargs={"n_gpu_layers": device, "verbose": True},
                    generation_kwargs={"max_tokens": 128, "temperature": temperature},
                )
            except Exception as e:
                logger.error(f"Failed to load gemma-7b: {e}")
                raise ValueError(f"Failed to load gemma-7b: {e}")

        # ------ Llama Model (3.1-8B) ------
        # --------------------------

        elif model_name == "llama3.1-8b":
            """
            llama-3.1-8b
            -------------

                - Model Type: LLaMA (Large Language Model Meta AI)

                - Parameters: 8 billion (8B)

                - Description:
                    Llama-3.1-8B is an improved version of the Llama-3 series, offering
                    a good balance between performance and resource efficiency.

                    It is suitable for a wide range of tasks, including conversational AI,
                    text generation, and instruction-following tasks.
            """

            logger.info("Loading llama3.1-8b model locally")
            try:
                llm = LlamaCppGenerator(
                    model=hf_hub_download(**local_models["llama-3.1-8b"]),
                    n_ctx=0,
                    n_batch=512,
                    model_kwargs={"n_gpu_layers": device, "verbose": True},
                    generation_kwargs={"max_tokens": 128, "temperature": temperature},
                )
            except Exception as e:
                logger.error(f"Failed to load llama3.1-8b: {e}")
                raise ValueError(f"Failed to load llama3.1-8b: {e}")

        # ------ Mistral Model (7B) ------
        # --------------------------

        elif model_name == "mistral-7b":
            """
            mistral-7b
            -----------

                - Model Type: Transformer Based LLM by Mistral AI

                - Parameters: 7 billion (7B)

                - Description:
                    Mistral 7B is highly optimized for efficiency and performance,
                    often outperforming models of larger sizes.

                    mistral-7B often outperforms larger models like
                    GPT-3 in generative tasks and instruction-following capabilities.

                    It is ideal for production
                    scenarios where memory efficiency is critical.
            """

            logger.info("Loading mistral-7b model locally")
            try:
                llm = LlamaCppGenerator(
                    model=hf_hub_download(**local_models["mistral-7b"]),
                    n_ctx=0,
                    n_batch=512,
                    model_kwargs={"n_gpu_layers": device, "verbose": True},
                    generation_kwargs={"max_tokens": 128, "temperature": temperature},
                )
            except Exception as e:
                logger.error(f"Failed to load mistral-7b: {e}")
                raise ValueError(f"Failed to load mistral-7b: {e}")

        # ------ Pythia Model (70M) ------
        # --------------------------

        elif model_name == "pythia-70m":
            """
            pythia-70m
            -----------

                - Model type: Transformer Based LLM by Mistral AI

                - Parameters: 70 million (70M)

                - Description:
                    The Pythia model suite was deliberately designed to promote scientific
                    research on large language models, especially interpretability research.

                    It is a lightweight model designed for applications that need fast
                    inference and low memory usage, making it ideal for
                    resource-constrained environments or for rapid prototyping.
            """

            logger.info("Loading pythia-70m model locally")
            try:
                llm = LlamaCppGenerator(
                    model=hf_hub_download(**local_models["pythia-70m"]),
                    n_ctx=0,
                    n_batch=512,
                    model_kwargs={"n_gpu_layers": device, "verbose": True},
                    generation_kwargs={"max_tokens": 128, "temperature": temperature},
                )
            except Exception as e:
                logger.error(f"Failed to load pythia-70m: {e}")
                raise ValueError(f"Failed to load pythia-70m: {e}")

        # ------ Pythia Model (410M) ------
        # --------------------------

        elif model_name == "pythia-410m":
            """
            Pythia-410m
            ------------

                - Model type: Transformer Based LLM by Mistral AI

                - Parameters: 410 million (410M)

                - Description:
                    Pythia-410M strikes a balance between small model size and
                    slightly more powerful language generation capabilities.

                    It is useful for more complex tasks than the 160M model, such as
                    summarization, basic conversational agents, and text generation with
                    moderate context.

                    The model is small enough to be efficient in deployment but
                    large enough to handle moderately complex NLP tasks.
            """

            logger.info("Loading pythia-410m model locally")
            try:
                llm = LlamaCppGenerator(
                    model=hf_hub_download(**local_models["pythia-410m"]),
                    n_ctx=0,
                    n_batch=512,
                    model_kwargs={"n_gpu_layers": device, "verbose": True},
                    generation_kwargs={"max_tokens": 128, "temperature": temperature},
                )
            except Exception as e:
                logger.error(f"Failed to load pythia-410m: {e}")
                raise ValueError(f"Failed to load pythia-410m: {e}")

        # ------ Pythia Model (1B) ------
        # --------------------------

        elif model_name == "pythia-1b":
            """
            Pythia-1.0b
            ------------

                - Model type: Transformer Based LLM by Mistral AI

                - Parameters: 1 billion (1B)

                - Description:
                    Pythia-1.0B is designed for more demanding natural language processing
                    tasks, such as document summarization, intermediate-level text
                    generation, and conversational AI applications.

                    The size of the model allows it to retain more context, making it
                    well-suited for medium-sized tasks while still maintaining
                    computational efficiency.
            """

            logger.info("Loading pythia-1b model locally")
            try:
                llm = LlamaCppGenerator(
                    model=hf_hub_download(**local_models["pythia-1b"]),
                    n_ctx=0,
                    n_batch=512,
                    model_kwargs={"n_gpu_layers": device, "verbose": True},
                    generation_kwargs={"max_tokens": 128, "temperature": temperature},
                )
            except Exception as e:
                logger.error(f"Failed to load pythia-1b: {e}")
                raise ValueError(f"Failed to load pythia-1b: {e}")

        # ------ Pythia Model (1.4B) ------
        # --------------------------

        elif model_name == "pythia-1.4b":
            """
            Pythia-1.4b
            ------------

                - Model type: Transformer Based LLM by Mistral AI

                - Parameters: 1.4 billion (1.4B)

                - Description:
                    Pythia-1.4B provides improved performance over the 1.0B model,
                    handling tasks that require better understanding of language patterns,
                    such as more accurate text generation, paraphrasing, and
                    complex dialogue systems.

                    It’s a versatile model that balances both the need for higher
                    performance and the requirement for relatively low computational resources.
            """

            logger.info("Loading pythia-1.4b model locally")
            try:
                llm = LlamaCppGenerator(
                    model=hf_hub_download(**local_models["pythia-1.4b"]),
                    n_ctx=0,
                    n_batch=512,
                    model_kwargs={"n_gpu_layers": device, "verbose": True},
                    generation_kwargs={"max_tokens": 128, "temperature": temperature},
                )
            except Exception as e:
                logger.error(f"Failed to load pythia-1.4b: {e}")
                raise ValueError(f"Failed to load pythia-1.4b: {e}")

        # ------ Pythia Model (2.8B) ------
        # --------------------------

        elif model_name == "pythia-2.8b":
            """
            Pythia-2.8b
            ------------

                - Model type: Transformer Based LLM by Mistral AI

                - Parameters: 2.8 billion (2.8B)

                - Description:
                    Pythia-2.8B is a more powerful version capable of handling
                    complex NLP tasks with longer text inputs and more
                    nuanced responses.

                    This model is an excellent choice when accuracy and context
                    retention are crucial and resource constraints when compared
                    to very large models.
            """

            logger.info("Loading pythia-2.8b model locally")
            try:
                llm = LlamaCppGenerator(
                    model=hf_hub_download(**local_models["pythia-2.8b"]),
                    n_ctx=0,
                    n_batch=512,
                    model_kwargs={"n_gpu_layers": device, "verbose": True},
                    generation_kwargs={"max_tokens": 128, "temperature": temperature},
                )
            except Exception as e:
                logger.error(f"Failed to load pythia-2.8b: {e}")
                raise ValueError(f"Failed to load pythia-2.8b: {e}")

        # ------ Alpaca-LoRA Model (7B) ------
        # --------------------------

        elif model_name == "alpaca-lora-7b":
            """
            Alpaca-LoRA-7b
            ---------------

                - Model Type: LLaMA-based with Low-Rank Adaptation (LoRA).

                - Parameters: 7 billion (7B)

                - Description:
                    Alpaca-LoRA is based on LLaMA-7B, fine-tuned using Alpaca-LoRA
                    for instruction-following tasks.

                    It offers efficiency in fine-tuning, making it a great option for
                    lightweight applications with strong task-following performance.

            """

            logger.info("Loading alpaca-lora-7b model locally")
            try:
                llm = LlamaCppGenerator(
                    model=hf_hub_download(**local_models["alpaca-lora-7b"]),
                    n_ctx=0,
                    n_batch=512,
                    model_kwargs={"n_gpu_layers": device, "verbose": True},
                    generation_kwargs={"max_tokens": 128, "temperature": temperature},
                )
            except Exception as e:
                logger.error(f"Failed to load alpaca-lora-7b: {e}")
                raise ValueError(f"Failed to load alpaca-lora-7b: {e}")

        # --------- Information ------------ >
        # ADD MORE MODELS HERE ------->

        else:
            logger.error(f"Model {model_name} not found in the local model list.")
            raise ValueError(
                f"Model {model_name} is not recognized. Please check the model name."
            )

    except Exception as e:
        logger.error(
            f"An error occurred while loading the model {model_name}: {e} on get_model function..."
        )
        raise ValueError(
            f"An error occurred while loading the model {model_name}: {e} on get_model function..."
        )

    finally:
        logger.info(f"{model_name} successfully loaded.")
        return llm
