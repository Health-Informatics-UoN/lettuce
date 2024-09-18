import argparse
import logging
import time
from typing import List, Dict

from haystack import Pipeline
from haystack.components.routers import ConditionalRouter

from components.embeddings import Embeddings
from components.models import get_model
from components.prompt import Prompts
from tests.test_prompt_build import mock_rag_results


class llm_pipeline:
    """
    This class is used to generate a pipeline for the model
    """

    def __init__(
        self, opt: argparse.Namespace, logger: logging.Logger | None = None
    ) -> None:
        """
        Initializes the llm_pipeline class

        Parameters
        ----------
        opt: argparse.Namespace
            Namespace containing the options for the pipeline
        logger: logging.Logger|None
            Logger for the pipeline
        """
        self._opt = opt
        self._model_name = opt.llm_model
        self._logger = logger
        if "llama-3.1" in opt.llm_model:
            self._eot_token = "<|eot_id|>"
        else:
            self._eot_token = ""

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

        pipeline.add_component("prompt", Prompts(
            model_name=self._model_name,
            eot_token=self._eot_token
            ).get_prompt())
        self._logger.info(f"Prompt added to pipeline in {time.time()-start} seconds")
        start = time.time()

        llm = get_model(
            model_name=self._model_name,
            temperature=self._opt.temperature,
            logger=self._logger,
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
                embeddings_path=self._opt.embeddings_path,
                force_rebuild=self._opt.force_rebuild,
                embed_vocab=self._opt.embed_vocab,
                model_name=self._opt.embedding_model,
                search_kwargs=self._opt.embedding_search_kwargs
                )
        
        vec_embedder = vec_search.get_embedder()
        vec_retriever = vec_search.get_retriever()
        router = ConditionalRouter(routes=[
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
                "output_type": List[Dict]
            }
            ])
        llm = get_model(
            model_name=self._model_name,
            temperature=self._opt.temperature,
            logger=self._logger,
        )
        
        pipeline.add_component("query_embedder", vec_embedder)
        pipeline.add_component("retriever", vec_retriever)
        pipeline.add_component("router", router)
        pipeline.add_component("prompt", Prompts(
            model_name=self._model_name,
            prompt_type="top_n_RAG",
            eot_token=self._eot_token
            ).get_prompt())
        pipeline.add_component("llm", llm)

        pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
        pipeline.connect("retriever.documents", "router.vec_results")
        pipeline.connect("router.no_exact_match", "prompt.vec_results")
        pipeline.connect("prompt.prompt", "llm.prompt")

        return pipeline
