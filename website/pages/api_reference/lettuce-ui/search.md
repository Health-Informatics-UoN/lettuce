# Search

## Functions
### `_text_search`
```python
_text_search(
    search_term: str,
    domain: List[str] | None,
    vocabulary: List[str] | None,
    standard_concept: bool,
    valid_concept: bool,
    top_k: int,
) -> List[ConceptSuggestion]:
```

Run a lexical search for a search term with specified parameters

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| `search_term` | str | The string to search for |
| `domain` | List[str] | None | A list of domains to include in results. If None, then all domains are searched |
| `vocabulary` | List[str] | None = None | A list of vocabularies to include in results. If None, then all vocabularies are searched |
| `standard_concept` | bool = True, | If true, only standard concepts returned. Otherwise, non-standard concepts will be included. |
| `valid_concept` | bool = True | If true, only concepts without invalid_reason returned. Otherwise, all concepts will be included. |
| `top_k` | int | The number of results to return  |

#### Returns `List[ConceptSuggestion]`
The top_k results from the lexical search

### `_vector_search`
```python
_vector_search(
    search_term: str,
    domain: List[str] | None,
    vocabulary: List[str] | None,
    standard_concept:bool,
    valid_concept: bool,
    embeddings_model_name: str,
    top_k: int,
) -> List[ConceptSuggestion]:
```

Run a vector search for a search term with specified parameters

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| `search_term` | str | The string to search for |
| `domain` | List[str] | None | A list of domains to include in results. If None, then all domains are searched |
| `vocabulary` | List[str] | None = None | A list of vocabularies to include in results. If None, then all vocabularies are searched |
| `standard_concept` | bool = True, | If true, only standard concepts returned. Otherwise, non-standard concepts will be included. |
| `valid_concept` | bool = True | If true, only concepts without invalid_reason returned. Otherwise, all concepts will be included. |
| `embeddings_model_name` | str, | The short name for an embeddings model to match on the EmbeddingModelName enum |
| `top_k` | int | The number of results to return  |

#### Returns `List[ConceptSuggestion]`
The top_k results from the vector search

### `_ai_search`
```python
_ai_search(
   search_term: str,
   domain: List[str] | None,
   vocabulary: List[str] | None,
   standard_concept: bool,
   valid_concept: bool,
   embeddings_model_name: str,
   top_k: int,
   llm_name: str,
   llm_url: str,
   logger: Logger,
) -> List[ConceptSuggestion]:
```

Run an LLM-powered search for a search term with specified parameters

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| `search_term` | str | The string to search for |
| `domain` | List[str] | None | A list of domains to include in results. If None, then all domains are searched |
| `vocabulary` | List[str] | None = None | A list of vocabularies to include in results. If None, then all vocabularies are searched |
| `standard_concept` | bool = True, | If true, only standard concepts returned. Otherwise, non-standard concepts will be included. |
| `valid_concept` | bool = True | If true, only concepts without invalid_reason returned. Otherwise, all concepts will be included. |
| `embeddings_model_name` | str, | The short name for an embeddings model to match on the EmbeddingModelName enum |
| `top_k` | int | The number of results to return  |
| `llm_name` | str | The short name for an LLM to match either a recognised name on a server or the locally saved models |
| `llm_url` | str | The URL for an LLM server |
| `logger` | Logger | log your problems |

#### Returns `List[ConceptSuggestion]`
The top_k results from the LLM-powered search

### `search`
```python
search(
   search_term: str,
   domain: List[str] | None,
   vocabulary: List[str] | None,
   standard_concept: bool,
   valid_concept: bool,
   top_k: int,
   search_mode: Literal["text-search", "vector-search", "ai-search"],
   embeddings_model_name: str,
   llm_name: str,
   llm_url: str,
   logger: Logger,
) -> List[ConceptSuggestion]:
```

Run search for a search term with specified parameters

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| `search_term` | str | The string to search for |
| `domain` | List[str] | None | A list of domains to include in results. If None, then all domains are searched |
| `vocabulary` | List[str] | None = None | A list of vocabularies to include in results. If None, then all vocabularies are searched |
| `standard_concept` | bool = True, | If true, only standard concepts returned. Otherwise, non-standard concepts will be included. |
| `valid_concept` | bool = True | If true, only concepts without invalid_reason returned. Otherwise, all concepts will be included. |
| `embeddings_model_name` | str, | The short name for an embeddings model to match on the EmbeddingModelName enum |
| `llm_name` | str | The short name for an LLM to match either a recognised name on a server or the locally saved models |
| `llm_url` | str | The URL for an LLM server |
| `logger` | Logger | log your problems |

#### Returns `List[ConceptSuggestion]`
The top_k results from the search

### `search_and_store`
```python
search_and_store(
    search_term: str,
    domain: List[str],
    vocabulary: List[str],
    standard_concept: bool,
    valid_concept: bool,
    top_k: int,
    search_mode: Literal["text-search", "vector-search", "ai-search"],
    embeddings_model_name: str,
    llm_name: str,
    llm_url: str,
    logger: Logger,
    result_storer: Callable,
) -> None:
```

Run a search for a search term with the provided parameters, then store the result

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| `search_term` | str | The string to search for |
| `domain` | List[str] | None | A list of domains to include in results. If None, then all domains are searched |
| `vocabulary` | List[str] | None = None | A list of vocabularies to include in results. If None, then all vocabularies are searched |
| `standard_concept` | bool = True, | If true, only standard concepts returned. Otherwise, non-standard concepts will be included. |
| `valid_concept` | bool = True | If true, only concepts without invalid_reason returned. Otherwise, all concepts will be included. |
| `top_k` | int | The number of results to return |
| `search_mode` | `Literal["text-search", "vector-search", "ai-search"]` | Which search function to use |
| `embeddings_model_name` | str, | The short name for an embeddings model to match on the EmbeddingModelName enum |
| `llm_name` | str | The short name for an LLM to match either a recognised name on a server or the locally saved models |
| `llm_url` | str | The URL for an LLM server |
| `logger` | Logger | log your problems |
| `result_storer` | Callable | A function to store results |
