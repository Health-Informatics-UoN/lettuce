import pytest
from components.prompt import Prompts

@pytest.fixture
def llama_3_simple_prompt_builder():
    return Prompts(
            model_name="llama-3-8b",
            prompt_type="simple",
            ).get_prompt()

@pytest.fixture
def llama_3_rag_prompt_builder():
    return Prompts(
            model_name="llama-3-8b",
            prompt_type="top_n_RAG",
            ).get_prompt()

@pytest.fixture
def llama_3_1_simple_prompt_builder():
    return Prompts(
            model_name="llama-3.1-8b",
            prompt_type="simple",
            eot_token="<|eot_id|>"
            ).get_prompt()

@pytest.fixture
def llama_3_1_rag_prompt_builder():
    return Prompts(
            model_name="llama-3.1-8b",
            prompt_type="top_n_RAG",
            eot_token="<|eot_id|>"
            ).get_prompt()

@pytest.fixture
def mock_rag_results():
    return [
            {'content': 'apple'}
            ]

def test_simple_prompt_returned(llama_3_simple_prompt_builder):
    assert "banana" in llama_3_simple_prompt_builder.run(informal_name="banana")["prompt"]

def test_rag_prompt_returned(llama_3_rag_prompt_builder, mock_rag_results):
    result = llama_3_rag_prompt_builder.run(informal_name="banana", vec_results=mock_rag_results)["prompt"]
    assert "banana" in result
    assert "apple" in result

def test_simple_prompt_with_eot(llama_3_1_simple_prompt_builder):
    result = llama_3_1_simple_prompt_builder.run(informal_name="banana")["prompt"]
    assert "banana" in result
    assert "<|eot_id|>" in result

def test_rag_prompt_with_eot(llama_3_1_rag_prompt_builder, mock_rag_results):
    result = llama_3_1_rag_prompt_builder.run(informal_name="banana", vec_results=mock_rag_results)["prompt"]
    assert "banana" in result
    assert "apple" in result
    assert "<|eot_id|>" in result
