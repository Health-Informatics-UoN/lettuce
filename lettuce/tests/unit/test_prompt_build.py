import pytest
from components.prompt import Prompts


@pytest.fixture
def llama_3_simple_prompt_builder():
    return Prompts(
        prompt_type="simple",
    ).get_prompt()


@pytest.fixture
def llama_3_rag_prompt_builder():
    return Prompts(
        prompt_type="top_n_RAG",
    ).get_prompt()


@pytest.fixture
def llama_3_1_simple_prompt_builder():
    return Prompts(prompt_type="simple").get_prompt()


@pytest.fixture
def llama_3_1_rag_prompt_builder():
    return Prompts(
        prompt_type="top_n_RAG",
    ).get_prompt()


@pytest.fixture
def mock_rag_results():
    return [{"content": "apple"}]


# Informal name tests
def test_simple_prompt_returned(llama_3_simple_prompt_builder):
    assert (
        "banana" in llama_3_simple_prompt_builder.run(informal_name="banana")["prompt"]
    )


def test_rag_prompt_returned(llama_3_rag_prompt_builder, mock_rag_results):
    result = llama_3_rag_prompt_builder.run(
        informal_name="banana", vec_results=mock_rag_results
    )["prompt"]
    assert "banana" in result
    assert "apple" in result


# Domain tests
def test_no_domain_provided(llama_3_1_rag_prompt_builder):
    result = llama_3_1_rag_prompt_builder.run()["prompt"]
    assert "source term, along" in result

def test_single_domain(llama_3_1_rag_prompt_builder):
    result = llama_3_1_rag_prompt_builder.run(domain = ["Drug"])["prompt"]
    assert "Drug, along" in result

def test_multiple_domains(llama_3_1_rag_prompt_builder):
    result = llama_3_1_rag_prompt_builder.run(domain = ["Drug", "Condition", "Observation"])["prompt"]
    assert "Drug, Condition, or Observation, along" in result
