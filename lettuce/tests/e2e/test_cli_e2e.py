import ast  
import subprocess 
import time 
import pytest 


@pytest.fixture(autouse=True, scope="session")
def time_e2e_session():
    start = time.time()
    yield
    end = time.time()
    print(f"\nTotal E2E test duration: {end - start:.2f} seconds")


def run_lettuce_cli_command(cli_args: list[str]): 
    cmd = ["uv", "run", "--env-file", ".env", "lettuce-cli"]
    cmd.extend(cli_args)   
    return subprocess.run(
        cmd, 
        capture_output=True, 
        text=True 
    )


def parse_output(output_str: str): 
    def find_balanced_brackets(s, start):
        open_count = 0
        for i in range(start, len(s)):
            if s[i] == '[':
                open_count += 1
            elif s[i] == ']':
                open_count -= 1
                if open_count == 0:
                    return i + 1  
        return -1  
    output_str = output_str.strip()
    list_start = output_str.find('[{')
    list_end = find_balanced_brackets(output_str, list_start)
    if list_start >= 0 and list_end > list_start:
        json_str = output_str[list_start:list_end]
        return ast.literal_eval(json_str)
    raise ValueError(f"Could not find JSON list in output: {output_str}")


def test_basic_search():  
    result = run_lettuce_cli_command([
        "--no-vector_search", 
        "--no-use_llm", 
        "--vocabulary_id", 
        "RxNorm", 
        "--informal_names", 
        "acetaminophen"
    ])
    output = parse_output(result.stdout)
    omop_matches = output[0]["OMOP matches"]["CONCEPT"]
    assert result.returncode == 0 
    assert len(output) > 0 
    assert any(match["concept_name"] == "acetaminophen" for match in omop_matches)
    assert all(match["vocabulary_id"] == "RxNorm" for match in omop_matches)


def test_basic_search_with_multiple_search_terms(): 
    result = run_lettuce_cli_command([
        "--no-vector_search", 
        "--no-use_llm",   
        "--informal_names", 
        "acetaminophen", 
        "Gastroenteritis", 
        "Rum"
    ])
    output = parse_output(result.stdout)
    assert result.returncode == 0 
    assert len(output) == 3 
    for i, search_term in enumerate(["acetaminophen", "Gastroenteritis", "Rum"]): 
        omop_matches = output[i]["OMOP matches"]["CONCEPT"]
        assert any(match["concept_name"] == search_term for match in omop_matches)


@pytest.mark.skip(reason="Acetaminophen/omeprazole not in the embeddings table")
def test_vector_search(): 
    result = run_lettuce_cli_command([
        "--vector_search", 
        "--no-use_llm", 
        "--informal_names", 
        "omeprazole"
    ])
    output = parse_output(result.stdout)
    omop_matches = output[0]["OMOP matches"]["CONCEPT"]
    vector_search_results = output[0]["Vector Search Results"]
    breakpoint()
    assert result.returncode == 0 
    assert len(output) > 0 
    assert len(omop_matches) > 0 
    assert any(match["concept_name"] == "omeprazole" for match in omop_matches)
    assert len(vector_search_results) > 0 
    assert any(match["concept"] == "omeprazole" for match in vector_search_results)


def test_llm_pipeline(): 
    result = run_lettuce_cli_command([
        "--no-vector_search", 
        "--use_llm", 
        "--model_name", 
        "tinyllama-1.1b-chat", 
        "--informal_names", 
        "acetaminophen"
    ])
    output = parse_output(result.stdout)
    assert len(output) > 0
    assert output[0]["llm_answer"] == "Acetaminophen"


@pytest.mark.skip(reason="Acetaminophen/omeprazole not in the embeddings table")
def test_combined_vector_and_llm(): 
    result = run_lettuce_cli_command([
        "--vector_search", 
        "--use_llm", 
        "--model_name", 
        "tinyllama-1.1b-chat", 
        "--informal_names", 
        "acetaminophen"
    ])
    output = parse_output(result.stdout)
    omop_matches = output[0]["OMOP matches"]["CONCEPT"]
    vector_search_results = output[0]["Vector Search Results"]
    assert result.returncode == 0 
    assert len(output) > 0 
    assert len(omop_matches) > 0 
    assert output[0]["llm_answer"] == "Acetaminophen"
    assert len(vector_search_results) > 0 
    assert any(match["content"] == "Acetaminophen" for match in vector_search_results)
