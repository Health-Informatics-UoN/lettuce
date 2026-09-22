from logging import Logger
import time
from typing import List, Dict

from haystack import Pipeline
from haystack.components.generators import OpenAIGenerator
from haystack.components.routers import ConditionalRouter
from haystack_integrations.components.generators.ollama import OllamaGenerator

from components.embeddings import Embeddings, EmbeddingModelName
from components.prompt import Prompts
from options.pipeline_options import InferenceType
from options.base_options import BaseOptions

settings = BaseOptions()

if settings.inference_type == InferenceType.LLAMA_CPP:
    try:
        from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
    except ImportError:
        raise ImportError("To use a Llama.cpp generator you have to install one of the optional dependency groups. Consult the documentation for details.")
    type Generator = LlamaCppGenerator|OpenAIGenerator|OllamaGenerator
else:
    type Generator = OpenAIGenerator|OllamaGenerator


class LLMPipeline:
    """
    This class is used to generate a pipeline for the model
    """

    def __init__(
        self,
        llm: Generator,
        temperature: float,
        logger: Logger,
        inference_type: InferenceType = settings.inference_type,
        inference_url: str | None = settings.ollama_url,
        embed_vocab: list[str] | None = None,
        standard_concept: bool = False,
        embedding_model: EmbeddingModelName = settings.embedding_model,
        top_k: int=5,
        verbose_llm: bool = False,
    ) -> None:
        """
        Initializes the LLMPipeline class

        Parameters
        ----------
        llm_model: Generator
            A haystack generator connecting to an LLM

        temperature: float
            The temperature the LLM uses for generation

        logger: logging.Logger|None
            Logger for the pipeline

        embed_vocab: List[str] | None
            If a list of OMOP vocabulary_ids is provided, filters RAG results by those vocabularies.

        standard_concept: bool
            If true, restricts RAG results to standard concepts

        embedding_model: EmbeddingModel
            The model used to create embeddings.

        top_k: int
            The number of RAG results to return

        verbose_llm: bool
            Whether the LLM should report on its running or not
        """
        self._model = llm
        self._inference_type = inference_type
        self._url = inference_url
        self._logger = logger
        self._temperature = temperature
        self._embed_vocab = embed_vocab
        self._standard_concept = standard_concept
        self._embedding_model = embedding_model
        self._top_k=top_k
        self._verbose_llm=verbose_llm


    @property
    def llm(self): 
        return self._model 

    def get_simple_assistant(self) -> Pipeline:
        """
        Get a simple assistant pipeline that connects a prompt with an LLM

        Returns
        -------
        Pipeline
            The pipeline for the assistant
        """
        start = time.time()
        pipeline = Pipeline()
        self._logger.info(f"Pipeline initialized in {time.time()-start} seconds")
        start = time.time()

        pipeline.add_component(
            "prompt",
            Prompts().get_prompt(),
        )
        self._logger.info(f"Prompt added to pipeline in {time.time()-start} seconds")
        start = time.time()

        pipeline.add_component("llm", self._model)
        self._logger.info(f"LLM added to pipeline in {time.time()-start} seconds")
        start = time.time()

        pipeline.connect("prompt.prompt", "llm.prompt")
        self._logger.info(f"Pipeline connected in {time.time()-start} seconds")

        return pipeline

    def get_rag_assistant(self) -> Pipeline:
        """
        Get an assistant that uses vector search to populate a prompt for an LLM

        Returns
        -------
        Pipeline
            The pipeline for the assistant
        """
        start = time.time()
        pipeline = Pipeline()
        self._logger.info(f"Pipeline initialized in {time.time()-start} seconds")
        start = time.time()

        vec_search = Embeddings(
            embed_vocab=self._embed_vocab,
            standard_concept=self._standard_concept,
            model_name=self._embedding_model,
            top_k=self._top_k,
        )

        vec_embedder = vec_search.get_embedder()
        vec_retriever = vec_search.get_retriever()
        router = ConditionalRouter(
            routes=[
                {
                    "condition": "{{vec_results[0].score < 0.05}}",
                    "output": "{{vec_results}}",
                    "output_name": "exact_match",
                    "output_type": List[Dict],
                },
                {
                    "condition": "{{vec_results[0].score >=0.05}}",
                    "output": """
                    {%- for result in vec_results %}
                    concept name: {{ result.content }} (score: {{ (100 * (1-result.score))|round(2) }}%)
                    {% endfor %}
                    """,
                    "output_name": "no_exact_match",
                    "output_type": List[Dict],
                },
            ]
        )

        pipeline.add_component("query_embedder", vec_embedder)
        pipeline.add_component("retriever", vec_retriever)
        pipeline.add_component("router", router)
        pipeline.add_component(
            "prompt",
            Prompts(
                prompt_type="top_n_RAG",
            ).get_prompt(),
        )
        pipeline.add_component("llm", self._model)

        pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
        pipeline.connect("retriever.documents", "router.vec_results")
        pipeline.connect("router.no_exact_match", "prompt.vec_results")
        pipeline.connect("prompt.prompt", "llm.prompt")

        return pipeline
