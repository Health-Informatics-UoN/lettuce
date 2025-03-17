import os 
from logging import Logger
import time
from typing import List, Dict

from haystack import Pipeline
from haystack.components.routers import ConditionalRouter

from components.embeddings import Embeddings, EmbeddingModelName
from components.models import get_model
from components.prompt import Prompts
from options.pipeline_options import LLMModel


class LLMPipeline:
    """
    This class is used to generate a pipeline for the model
    """

    def __init__(
        self,
        llm_model: LLMModel,
        temperature: float,
        logger: Logger,
        embeddings_path: str = "",
        force_rebuild: bool = False,
        embed_vocab: list[str] = [],
        embedding_model: EmbeddingModelName = EmbeddingModelName.BGESMALL,
        embedding_search_kwargs: dict = {},
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

        embeddings_path: str
            A path for the embeddings database. If one is not found,
            it will be built, which takes a long time. This is built
            from concepts fetched from the OMOP database.

        force_rebuild: bool
            If true, the embeddings database will be rebuilt.

        embed_vocab: List[str]
            A list of OMOP vocabulary_ids. If the embeddings database is
            built, these will be the vocabularies used in the OMOP query.

        embedding_model: EmbeddingModel
            The model used to create embeddings.

        embedding_search_kwargs: dict
            kwargs for vector search.
        """
        self._model = llm_model
        self._logger = logger
        self._temperature = temperature
        self._embeddings_path = embeddings_path
        self._force_rebuild = force_rebuild
        self._embed_vocab = embed_vocab
        self._embedding_model = embedding_model
        self._embedding_search_kwargs = embedding_search_kwargs

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

        path_to_local_model_weights = os.getenv("LOCAL_LLM")
        llm = get_model(
            model=self._model,
            temperature=self._temperature,
            logger=self._logger,
            path_to_local_weights=path_to_local_model_weights
        )
        pipeline.add_component("llm", llm)
        self._logger.info(f"LLM added to pipeline in {time.time()-start} seconds")
        start = time.time()

        pipeline.connect("prompt.prompt", "llm.prompt")
        self._logger.info(f"Pipeline connected in {time.time()-start} seconds")

        return pipeline

    def get_rag_assistant(self) -> Pipeline:
        print("Real get_rag_assistant called")
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
            embeddings_path=self._embeddings_path,
            force_rebuild=self._force_rebuild,
            embed_vocab=self._embed_vocab,
            model_name=self._embedding_model,
            search_kwargs=self._embedding_search_kwargs,
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

        path_to_local_model_weights = os.getenv("LOCAL_LLM")
        llm = get_model(
            model=self._model,
            temperature=self._temperature,
            logger=self._logger,
            path_to_local_weights=path_to_local_model_weights
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
