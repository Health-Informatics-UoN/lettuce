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

        # ------ Kuchiki Model (7B) ------
        # --------------------------

        elif model_name == "kuchiki-l2-7b":
            """
            kuchiki-l2-7b
            -------------

                - Model Type: LLaMA based by Kuchiki Research

                - Parameters: 7 billion (7B)

                - Description:
                    The Kuchiki-L2-7B model is a hybrid language model built on Llama-2,
                    blending Nous Hermes, Airoboros, and LimaRP to excel in
                    instruction-following and role-playing tasks.
            """

            logger.info("Loading kuchiki-l2-7b model locally")
            try:
                llm = LlamaCppGenerator(
                    model=hf_hub_download(**local_models["kuchiki-l2-7b"]),
                    n_ctx=0,
                    n_batch=512,
                    model_kwargs={"n_gpu_layers": device, "verbose": True},
                    generation_kwargs={"max_tokens": 128, "temperature": temperature},
                )
            except Exception as e:
                logger.error(f"Failed to load kuchiki-l2-7b: {e}")
                raise ValueError(f"Failed to load kuchiki-l2-7b: {e}")

        # ------ TinyLlama Model (1.1B) ------
        # --------------------------

        elif model_name == "tinyllama-1.1b-chat":
            """
            tinyllama-1.1b-chat
            -------------------

                - Model Type: LLaMA (Large Language Model Meta AI)

                - Parameters: 1.1 billion (1.1B)

                - Description:

                    The TinyLlama-1.1B-Chat model is a compact version
                    of the Llama-2-7B model, optimized for chat and
                    general text generation tasks.
            """

            logger.info("Loading tinyllama-1.1b-chat model locally")
            try:
                llm = LlamaCppGenerator(
                    model=hf_hub_download(**local_models["tinyllama-1.1b-chat"]),
                    n_ctx=0,
                    n_batch=512,
                    model_kwargs={"n_gpu_layers": device, "verbose": True},
                    generation_kwargs={"max_tokens": 128, "temperature": temperature},
                )
            except Exception as e:
                logger.error(f"Failed to load tinyllama-1.1b-chat: {e}")
                raise ValueError(f"Failed to load tinyllama-1.1b-chat: {e}")

        # ------ BioMistral Model (7B) ------
        # --------------------------

        elif model_name == "biomistral-7b":
            """
            bioMistral-7B
            -------------

                - Model Type: Transformer Based LLM by Mistral AI

                - Parameters: 7 billion (7B)

                - Description:
                    BioMistral-7B-GGUF is a specialized version of the Mistral-7B model,
                    optimized for bioinformatics and life sciences tasks.
            """

            logger.info("Loading bioMistral-7B-GGUF model locally")
            try:
                llm = LlamaCppGenerator(
                    model=hf_hub_download(**local_models["biomistral-7b"]),
                    n_ctx=0,
                    n_batch=512,
                    model_kwargs={"n_gpu_layers": device, "verbose": True},
                    generation_kwargs={"max_tokens": 128, "temperature": temperature},
                )
            except Exception as e:
                logger.error(f"Failed to load bioMistral-7B-GGUF: {e}")
                raise ValueError(f"Failed to load bioMistral-7B-GGUF: {e}")

        # ------ Qwen2.5-3B-Instruct-GGUF ------
        # --------------------------

        elif model_name == "qwen2.5-3b-instruct":
            """
            Qwen2.5-3B-Instruct-GGUF
            ------------------------

                - Model Type: Transformer Based LLM by Qwen

                - Parameters: 3 billion (3B)

                - Description:
                    Qwen2.5-3B-Instruct-GGUF is a specialized version of the Qwen2.5-3B model,
                    optimized for instruction-following tasks.
            """

            logger.info("Loading Qwen2.5-3B-Instruct-GGUF model locally")
            try:
                llm = LlamaCppGenerator(
                    model=hf_hub_download(**local_models["qwen2.5-3b-instruct"]),
                    n_ctx=0,
                    n_batch=512,
                    model_kwargs={"n_gpu_layers": device, "verbose": True},
                    generation_kwargs={"max_tokens": 128, "temperature": temperature},
                )
            except Exception as e:
                logger.error(f"Failed to load Qwen2.5-3B-Instruct-GGUF: {e}")
                raise ValueError(f"Failed to load Qwen2.5-3B-Instruct-GGUF: {e}")

        # ------ Airoboros-3B ------
        # --------------------------

        elif model_name == "airoboros-3b":
            """
            airoboros-3b
            ------------

                - Model Type: Transformer Based LLM by Afrideva

                - Parameters: 3 billion (3B)

                - Description:
                    Airoboros-3B is a compact model optimized for general-purpose
                    text generation tasks.
            """

            logger.info("Loading airoboros-3b model locally")
            try:
                llm = LlamaCppGenerator(
                    model=hf_hub_download(**local_models["airoboros-3b"]),
                    n_ctx=0,
                    n_batch=512,
                    model_kwargs={"n_gpu_layers": device, "verbose": True},
                    generation_kwargs={"max_tokens": 128, "temperature": temperature},
                )
            except Exception as e:
                logger.error(f"Failed to load airoboros-3b: {e}")
                raise ValueError(f"Failed to load airoboros-3b: {e}")

        # ------ Medicine Chat ------
        # --------------------------
        elif model_name == "medicine-chat":
            """
            medicine-chat
            -------------

                - Model Type: LLaMA (Large Language Model Meta AI)

                - Parameters: 7 billion (7B)

                - Description:
                    The Medicine-Chat model is a specialized version of the Llama-2-7B model,
                    optimized for medical and healthcare-related text generation tasks.
            """

            logger.info("Loading medicine-chat model locally")
            try:
                llm = LlamaCppGenerator(
                    model=hf_hub_download(**local_models["medicine-chat"]),
                    n_ctx=0,
                    n_batch=512,
                    model_kwargs={"n_gpu_layers": device, "verbose": True},
                    generation_kwargs={"max_tokens": 128, "temperature": temperature},
                )
            except Exception as e:
                logger.error(f"Failed to load medicine-chat: {e}")
                raise ValueError(f"Failed to load medicine-chat: {e}")

        # ------ Medicine LLM 13B ------
        # --------------------------

        elif model_name == "medicine-llm-13b":
            """
            medicine-llm-13b
            ----------------

                - Model Type: LLaMA (Large Language Model Meta AI)

                - Parameters: 13 billion (13B)

                - Description:
                    The Medicine-LLM-13B model is a specialized version of the Llama-3-8B model,
                    optimized for medical and healthcare-related text generation tasks.
            """

            logger.info("Loading medicine-llm-13b model locally")
            try:
                llm = LlamaCppGenerator(
                    model=hf_hub_download(**local_models["medicine-llm-13b"]),
                    n_ctx=0,
                    n_batch=512,
                    model_kwargs={"n_gpu_layers": device, "verbose": True},
                    generation_kwargs={"max_tokens": 128, "temperature": temperature},
                )
            except Exception as e:
                logger.error(f"Failed to load medicine-llm-13b: {e}")
                raise ValueError(f"Failed to load medicine-llm-13b: {e}")

        # ------ Med-Llama Models ------
        # --------------------------

        elif model_name == "med-llama-3-8b-v1":
            """
            med-llama-3-8b-v1
            -----------------

                - Model Type: LLaMA (Large Language Model Meta AI)

                - Parameters: 3.8 billion (3.8B)

                - Description:
                    The Med-Llama-3-8B-v1 model is a specialized version of the Llama-3-8B model,
                    optimized for medical and healthcare-related text generation tasks.
            """

            logger.info("Loading med-llama-3-8b-v1 model locally")
            try:
                llm = LlamaCppGenerator(
                    model=hf_hub_download(**local_models["med-llama-3-8b-v1"]),
                    n_ctx=0,
                    n_batch=512,
                    model_kwargs={"n_gpu_layers": device, "verbose": True},
                    generation_kwargs={"max_tokens": 128, "temperature": temperature},
                )
            except Exception as e:
                logger.error(f"Failed to load med-llama-3-8b-v1: {e}")
                raise ValueError(f"Failed to load med-llama-3-8b-v1: {e}")

        # ------ Med-Llama Models ------
        # --------------------------

        elif model_name == "med-llama-3-8b-v2":
            """
            med-llama-3-8b-v2
            -----------------

                - Model Type: LLaMA (Large Language Model Meta AI)

                - Parameters: 3.8 billion (3.8B)

                - Description:
                    The Med-Llama-3-8B-v2 model is a specialized version of the Llama-3-8B model,
                    optimized for medical and healthcare-related text generation tasks.
            """

            logger.info("Loading med-llama-3-8b-v2 model locally")
            try:
                llm = LlamaCppGenerator(
                    model=hf_hub_download(**local_models["med-llama-3-8b-v2"]),
                    n_ctx=0,
                    n_batch=512,
                    model_kwargs={"n_gpu_layers": device, "verbose": True},
                    generation_kwargs={"max_tokens": 128, "temperature": temperature},
                )
            except Exception as e:
                logger.error(f"Failed to load med-llama-3-8b-v2: {e}")
                raise ValueError(f"Failed to load med-llama-3-8b-v2: {e}")

        # ------ Med-Llama Models ------
        # --------------------------

        elif model_name == "med-llama-3-8b-v3":
            """
            med-llama-3-8b-v3
            -----------------

                - Model Type: LLaMA (Large Language Model Meta AI)

                - Parameters: 3.8 billion (3.8B)

                - Description:
                    The Med-Llama-3-8B-v3 model is a specialized version of the Llama-3-8B model,
                    optimized for medical and healthcare-related text generation tasks.
            """

            logger.info("Loading med-llama-3-8b-v3 model locally")
            try:
                llm = LlamaCppGenerator(
                    model=hf_hub_download(**local_models["med-llama-3-8b-v3"]),
                    n_ctx=0,
                    n_batch=512,
                    model_kwargs={"n_gpu_layers": device, "verbose": True},
                    generation_kwargs={"max_tokens": 128, "temperature": temperature},
                )
            except Exception as e:
                logger.error(f"Failed to load med-llama-3-8b-v3: {e}")
                raise ValueError(f"Failed to load med-llama-3-8b-v3: {e}")

        # ------ Med-Llama Models ------
        # --------------------------

        elif model_name == "med-llama-3-8b-v4":
            """
            med-llama-3-8b-v4
            -----------------

                - Model Type: LLaMA (Large Language Model Meta AI)

                - Parameters: 3.8 billion (3.8B)

                - Description:
                    The Med-Llama-3-8B-v4 model is a specialized version of the Llama-3-8B model,
                    optimized for medical and healthcare-related text generation tasks.
            """

            logger.info("Loading med-llama-3-8b-v4 model locally")
            try:
                llm = LlamaCppGenerator(
                    model=hf_hub_download(**local_models["med-llama-3-8b-v4"]),
                    n_ctx=0,
                    n_batch=512,
                    model_kwargs={"n_gpu_layers": device, "verbose": True},
                    generation_kwargs={"max_tokens": 128, "temperature": temperature},
                )
            except Exception as e:
                logger.error(f"Failed to load med-llama-3-8b-v4: {e}")
                raise ValueError(f"Failed to load med-llama-3-8b-v4: {e}")

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
