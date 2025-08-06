from enum import Enum
from pydantic import BaseModel
from components.embeddings import EmbeddingModelName


class LLMModel(str, Enum):
    """
    This enum holds the names of the different models the assistant can use
    """

    GPT_3_5_TURBO = "gpt-3.5-turbo-0125"
    GPT_4 = "gpt-4"
    LLAMA_2_7B = "llama-2-7b-chat"
    LLAMA_3_8B = "llama-3-8b"
    LLAMA_3_70B = "llama-3-70b"
    GEMMA_7B = "gemma-7b"
    LLAMA_3_1_8B = "llama-3.1-8b"
    LLAMA_3_2_3B = "llama-3.2-3b"
    MISTRAL_7B = "mistral-7b"
    KUCHIKI_L2_7B = "kuchiki-l2-7b"
    TINYLLAMA_1_1B_CHAT = "tinyllama-1.1b-chat"
    BIOMISTRAL_7B = "biomistral-7b"
    QWEN2_5_3B_INSTRUCT = "qwen2.5-3b-instruct"
    AIROBOROS_3B = "airoboros-3b"
    MEDICINE_CHAT = "medicine-chat"
    MEDICINE_LLM_13B = "medicine-llm-13b"
    MED_LLAMA_3_8B_V1 = "med-llama-3-8b-v1"
    MED_LLAMA_3_8B_V2 = "med-llama-3-8b-v2"
    MED_LLAMA_3_8B_V3 = "med-llama-3-8b-v3"
    MED_LLAMA_3_8B_V4 = "med-llama-3-8b-v4"

    def get_eot_token(self) -> str:
        if self.value in [
            "llama-3.1-8b",
        ]:
            return "<|eot_id|>"
        return ""


class PipelineOptions(BaseModel):
    """
    This class holds the options available to the
    Llettuce pipeline.

    These are all the options in the BaseOptions parser.
    The defaults provided here match the default options in
    BaseOptions. Using a pydantic model means FastAPI
    can take these as input in the API request.

    Attributes
    ----------
    llm_model: LLMModel
        The name of the LLM used in the pipeline. The permitted
        values are the possibilities in the LLMModel enum.

    temperature: float
        Temperature supplied to the LLM that tunes the
        variability of responses.

    concept_ancestor: bool (Defaults to false)
        If true, the concept_ancestor table of the OMOP vocabularies
        is queried for the results of an OMOP search.

    concept_relationship: bool (Defaults to false)
        If true, the concept_relationship table of the OMOP vocabularies
        is queried for the results of an OMOP search.

    concept_synonym: bool (Defaults to false)
        If true, the concept_synonym table of the OMOP vocabularies
        is queried when OMOP concepts are fetched.

    search_threshold: int
        The threshold on fuzzy string matching for returned results.

    max_separation_descendant: int
        The maximum separation to search for concept descendants.

    max_separation_ancestor: int
        The maximum separation to search for concept ancestors
    """

    vocabulary_id: list[str] = ["RxNorm"]
    standard_concept: bool = True
    concept_ancestor: bool = False
    concept_relationship: bool = False
    concept_synonym: bool = False
    search_threshold: int = 80
    max_separation_descendants: int = 1
    max_separation_ancestor: int = 1
    embed_vocab: list[str] = ["RxNorm", "RxNorm Extension"]
    embeddings_top_k: int = 5
