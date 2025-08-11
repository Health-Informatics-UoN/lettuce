from enum import Enum
from pydantic import BaseModel

class InferenceType(str, Enum):
    """
    This enum holds the different ways that users can perform inference
    """

    OPEN_AI = "OpenAI"
    OLLAMA = "Ollama"
    LLAMA_CPP = "llama-cpp-python"

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
    GEMMA_3N_E4B = "gemma3n:e4b"

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

class EmbeddingModelName(str, Enum):
    """
    This class enumerates the embedding models we
    have the download details for.

    The models are:
    """

    BGESMALL = "BGESMALL"
    MINILM = "MINILM"
    GTR_T5_BASE = "gtr-t5-base"
    GTR_T5_LARGE = "gtr-t5-large"
    E5_BASE = "e5-base"
    E5_LARGE = "e5-large"
    DISTILBERT_BASE_UNCASED = "distilbert-base-uncased"
    DISTILUSE_BASE_MULTILINGUAL = "distiluse-base-multilingual-cased-v1"
    CONTRIEVER = "contriever"


class EmbeddingModelInfo(BaseModel):
    """
    A simple class to hold the information for embeddings models
    """

    path: str
    dimensions: int


class EmbeddingModel(BaseModel):
    """
    A class to match the name of an embeddings model with the
    details required to download and use it.
    """

    name: EmbeddingModelName
    info: EmbeddingModelInfo


EMBEDDING_MODELS = {
    # ------ Bidirectional Gated Encoder  ------- >
    EmbeddingModelName.BGESMALL: EmbeddingModelInfo(
        path="BAAI/bge-small-en-v1.5", dimensions=384
    ),
    # ------ SBERT (Sentence-BERT) ------- >
    EmbeddingModelName.MINILM: EmbeddingModelInfo(
        path="sentence-transformers/all-MiniLM-L6-v2", dimensions=384
    ),
    # ------ Generalizable T5 Retrieval ------- >
    EmbeddingModelName.GTR_T5_BASE: EmbeddingModelInfo(
        path="google/gtr-t5-base", dimensions=768
    ),
    # ------ Generalizable T5 Retrieval ------- >
    EmbeddingModelName.GTR_T5_LARGE: EmbeddingModelInfo(
    path="google/gtr-t5-large", dimensions=1024
    ),
    # ------ Embedding Models for Search Engines ------- >
    EmbeddingModelName.E5_BASE: EmbeddingModelInfo(
        path="microsoft/e5-base", dimensions=768
    ),
    # ------ Embedding Models for Search Engines ------- >
    EmbeddingModelName.E5_LARGE: EmbeddingModelInfo(
        path="microsoft/e5-large", dimensions=1024
    ),
    # ------ DistilBERT ------- >
    EmbeddingModelName.DISTILBERT_BASE_UNCASED: EmbeddingModelInfo(
        path="distilbert-base-uncased", dimensions=768
    ),
    # ------ distiluse-base-multilingual-cased-v1 ------- >
    EmbeddingModelName.DISTILUSE_BASE_MULTILINGUAL: EmbeddingModelInfo(
        path="sentence-transformers/distiluse-base-multilingual-cased-v1",
        dimensions=512,
    ),
    # ------ Contriever ------- >
    EmbeddingModelName.CONTRIEVER: EmbeddingModelInfo(
        path="facebook/contriever", dimensions=768
    ),
}
