import argparse
import logging
import time

from haystack import Pipeline

from components.models import get_model
from components.prompt import Prompts


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

        pipeline.add_component("prompt", Prompts(self._model_name).get_prompt())
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
        start = time.time()

        return pipeline
