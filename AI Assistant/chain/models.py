import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from langchain_community.llms import GPT4All, LlamaCpp
from langchain_openai import ChatOpenAI


def get_model(
    hub: str, model_name: str, temperature: float = 0.7
) -> ChatOpenAI | LlamaCpp | GPT4All:
    """
    Get the model

    Parameters:
    ----------
    hub: str
        The hub to use
    model_name: str
        The model name to use
    temperature: float
        The temperature to use

    Returns:
    -------
    Model
        The model
    """

    if hub.lower() == "openai":
        return ChatOpenAI(model=model_name, temperature=temperature)

    elif hub.lower() == "llamacpp":
        if model_name.lower() == "llama-2-7b":
            """
            [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-hf)
            [GGUF format](https://huggingface.co/TheBloke/Llama-2-7B-GGUF)
            """
            repo = "TheBloke/Llama-2-7B-GGUF"
            filename = "llama-2-7b.Q4_0.gguf"  # Options: llama-2-7b.Q4_0.gguf, llama-2-7b.Q5_0.gguf, llama-2-7b.Q8_0.gguf

        elif model_name.lower() == "llama-2-7b-chat":
            """
            [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
            [GGUF format](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)
            """
            repo = "TheBloke/Llama-2-7B-Chat-GGUF"
            filename = "llama-2-7b-chat.Q4_0.gguf"  # Options: llama-2-7b-chat.Q4_0.gguf, llama-2-7b-chat.Q5_0.gguf, llama-2-7b-chat.Q8_0.gguf

        elif model_name.lower() == "llama-2-13b":
            """
            [Llama-2](https://huggingface.co/meta-llama/Llama-2-13b-hf)
            [GGUF format](https://huggingface.co/TheBloke/Llama-2-13B-GGUF)
            """
            repo = "TheBloke/Llama-2-13B-GGUF"
            filename = "llama-2-13b.Q4_0.gguf"  # Options: llama-2-13b.Q4_0.gguf, llama-2-13b.Q5_0.gguf, llama-2-13b.Q8_0.gguf

        elif model_name.lower() == "llama-2-13b-chat":
            """
            [Llama-2](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
            [GGUF format](https://huggingface.co/TheBloke/Llama-2-13B-Chat-GGUF)
            """
            repo = "TheBloke/Llama-2-13B-Chat-GGUF"
            filename = "llama-2-13b-chat.Q4_0.gguf"  # Options: llama-2-13b-chat.Q4_0.gguf, llama-2-13b-chat.Q5_0.gguf, llama-2-13b-chat.Q8_0.gguf

        elif model_name.lower() == "llama-2-70b-chat":
            """
            [Llama-2](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)
            [GGUF format](https://huggingface.co/TheBloke/Llama-2-70B-Chat-GGUF)
            """
            repo = "TheBloke/Llama-2-70B-Chat-GGUF"
            filename = "llama-2-70b-chat.Q4_0.gguf"  # Options: llama-2-70b-chat.Q4_0.gguf, llama-2-70b-chat.Q5_0.gguf

        else:
            raise ValueError(f"Invalid model: {hub}/{model_name}")

        n_gpu_layers = -1 if torch.cuda.is_available() else 0
        gguf_model = hf_hub_download(
            repo_id=repo,
            filename=filename,
        )
        return LlamaCpp(
            model_path=gguf_model,
            n_gpu_layers=n_gpu_layers,
            temperature=temperature,
            n_ctx=0,  # Text context, 0 = from model
            n_batch=512,
            max_tokens=2048,
            f16_kv=True,
            verbose=True,
        )

    elif hub.lower() == "gpt4all":
        if model_name.lower() == "mistral-7b-openorca":
            model = "mistral-7b-openorca.gguf2.Q4_0.gguf"
        elif model_name.lower() == "mistral-7b-instruct":
            model = "mistral-7b-instruct-v0.1.Q4_0.gguf"
        elif model_name.lower() == "gpt4all-falcon-newbpe":
            model = "gpt4all-falcon-newbpe-q4_0.gguf"
        device = "gpu" if torch.cuda.is_available() else "cpu"
        return GPT4All(
            model=model,
            temp=temperature,
            n_batch=512,
            n_predict=2048,
            verbose=True,
            allow_download=True,
            device=device,
        )

    else:
        raise ValueError(f"Invalid hub: {hub}")
