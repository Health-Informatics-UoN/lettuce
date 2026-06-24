from .connect import get_model, connect_to_ollama, connect_to_openai
from .local_models import get_local_weights, download_model_from_huggingface

__all__ = [
        "get_model",
        "connect_to_openai",
        "connect_to_ollama",
        "get_local_weights",
        "download_model_from_huggingface"
        ]
