from logging import Logger
import time
from typing import List, Dict

from haystack import Pipeline
from haystack.components.routers import ConditionalRouter

from components.embeddings import Embeddings, EmbeddingModelName
from components.models import get_model
from components.prompt import Prompts
from options.pipeline_options import LLMModel
from options.base_options import BaseOptions

settings = BaseOptions()

class LLMPipeline:
    """
    This class is used to generate a pipeline for the model
    """

    def __init__(
        self,
        llm_model: LLMModel,
        temperature: float,
        logger: Logger,
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
        llm_model: LLMModel
            The choice of LLM to run the pipeline

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
        self._model = llm_model
        self._logger = logger
        self._temperature = temperature
        self._embed_vocab = embed_vocab
        self._standard_concept = standard_concept
        self._embedding_model = embedding_model
        self._top_k=top_k
        self._verbose_llm=verbose_llm

    @property
    def llm_model(self): 
        return self._model 

    @llm_model.setter
    def llm_model(self, value): 
        self._model = value 

    @property
    def llm_model(self): 
        return self._model 

    @llm_model.setter
    def llm_model(self, value): 
        self._model = value 

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
            Prompts(model=self._model).get_prompt(),
        )
        self._logger.info(f"Prompt added to pipeline in {time.time()-start} seconds")
        start = time.time()

        llm = get_model(
            model=self._model,
            inference_type=settings.inference_type,
            temperature=self._temperature,
            url=settings.ollama_url,
            logger=self._logger,
            path_to_local_weights=settings.local_llm,
            verbose=self._verbose_llm,
        )
        pipeline.add_component("llm", llm)
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
                    "condition": "{{vec_results[0].score > 0.95}}",
                    "output": "{{vec_results}}",
                    "output_name": "exact_match",
                    "output_type": List[Dict],
                },
                {
                    "condition": "{{vec_results[0].score <=0.95}}",
                    "output": "{{vec_results}}",
                    "output_name": "no_exact_match",
                    "output_type": List[Dict],
                },
            ]
        )

        llm = get_model(
            model=self._model,
            inference_type=settings.inference_type,
            url=settings.ollama_url,
            temperature=self._temperature,
            logger=self._logger,
            path_to_local_weights=settings.local_llm,
            verbose=self._verbose_llm,
        )

        pipeline.add_component("query_embedder", vec_embedder)
        pipeline.add_component("retriever", vec_retriever)
        pipeline.add_component("router", router)
        pipeline.add_component(
            "prompt",
            Prompts(
                model=self._model,
                prompt_type="top_n_RAG",
            ).get_prompt(),
        )
        pipeline.add_component("llm", llm)

        pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
        pipeline.connect("retriever.documents", "router.vec_results")
        pipeline.connect("router.no_exact_match", "prompt.vec_results")
        pipeline.connect("prompt.prompt", "llm.prompt")

        return pipeline
