from enum import Enum
from pydantic import BaseModel
from components.embeddings import EmbeddingModelName
from options.base_options import BaseOptions


class LLMModel(str, Enum):
    """
    This enum holds the names of the different models the assistant can use
    """

    GPT_3_5_TURBO = "gpt-3.5-turbo-0125"
    GPT_4 = "gpt-4"
    LLAMA_2_7B = "llama-2-7B-chat"
    LLAMA_3_8B = "llama-3-8b"
    LLAMA_3_70B = "llama-3-70b"
    GEMMA_7B = "gemma-7b"
    LLAMA_3_1_8B = "llama-3.1-8b"
    MISTRAL_7B = "mistral-7b"
    PYTHIA_70M = "pythia-70m"
    PYTHIA_410M = "pythia-410m"
    PYTHIA_1B = "pythia-1b"
    PYTHIA_1_4B = "pythia-1.4b"
    PYTHIA_2_8B = "pythia-2.8b"
    ALPACA_LORA_7B = "alpaca-lora-7b"


class PipelineOptions(BaseModel):
    """
    This class holds the options available to the Llettuce pipeline

    These are all the options in the BaseOptions parser. The defaults provided here match the default options in BaseOptions. Using a pydantic model means FastAPI can take these as input in the API request

    Attributes
    ----------
    llm_model: LLMModel
        The name of the LLM used in the pipeline. The permitted values are the possibilities in the LLMModel enum
    temperature: float
        Temperature supplied to the LLM that tunes the variability of responses
    concept_ancestor: bool
        If true, the concept_ancestor table of the OMOP vocabularies is queried for the results of an OMOP search. Defaults to false
    concept_relationship: bool
        If true, the concept_relationship table of the OMOP vocabularies is queried for the results of an OMOP search. Defaults to false
    concept_synonym: bool
        If true, the concept_synonym table of the OMOP vocabularies is queried when OMOP concepts are fetched. Defaults to false
    search_threshold: int
        The threshold on fuzzy string matching for returned results
    max_separation_descendant: int
        The maximum separation to search for concept descendants
    max_separation_ancestor: int
        The maximum separation to search for concept ancestors
    """

    llm_model: LLMModel = LLMModel.LLAMA_3_1_8B
    temperature: float = 0
    vocabulary_id: str = "RxNorm"  # TODO: make multiples possible
    concept_ancestor: bool = False
    concept_relationship: bool = False
    concept_synonym: bool = False
    search_threshold: int = 80
    max_separation_descendants: int = 1
    max_separation_ancestor: int = 1
    embeddings_path: str = "concept_embeddings.qdrant"
    force_rebuild: bool = False
    embed_vocab: list[str] = ["RxNorm", "RxNorm Extension"]
    embedding_model: EmbeddingModelName = EmbeddingModelName.BGESMALL
    embedding_search_kwargs: dict = {}


def parse_pipeline_args(base_options: BaseOptions, options: PipelineOptions) -> None:
    """
    Use the values of a PipelineOptions object to override defaults

    Parameters
    ----------
    base_options: BaseOptions
        The base options from the command-line application
    options: PipelineOptions
        Overrides from an API request

    Returns
    -------
    None
    """
    base_options._parser.set_defaults(
        llm_model=options.llm_model.value,
        temperature=options.temperature,
        vocabulary_id=options.vocabulary_id,
        concept_ancestor="y" if options.concept_ancestor else "n",
        concept_relationship="y" if options.concept_relationship else "n",
        concept_synonym="y" if options.concept_synonym else "n",
        search_threshold=options.search_threshold,
        max_separation_descendants=options.max_separation_descendants,
        max_separation_ancestor=options.max_separation_ancestor,
    )
