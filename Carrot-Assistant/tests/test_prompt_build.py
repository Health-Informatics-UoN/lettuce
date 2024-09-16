import pytest
from components.prompt import Prompts

@pytest.fixture
def llama_3_simple_prompt_builder():
    return Prompts(
            model_name="llama-3-8b",
            prompt_type="simple",
            )

@pytest.fixture
def llama_3_rag_prompt_builder():
    return Prompts(
            model_name="llama-3-8b",
            prompt_type="top_n_RAG",
            )

@pytest.fixture
def llama_3_1_simple_prompt_builder():
    return Prompts
