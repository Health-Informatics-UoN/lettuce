from typing import Dict
from evaluation.evaltypes import (
    SingleResultPipelineTest,
    SingleResultMetric,
)
from evaluation.pipelines import LLMPipeline, RAGPipeline


class LLMPipelineTest(SingleResultPipelineTest):
    """
    This class provides a pipeline test for LLM pipelines that return a single result
    """

    def __init__(
        self,
        name: str,
        pipeline: LLMPipeline,
        metrics: list[SingleResultMetric],
    ):
        """
        Initialises the LLMPipelineTest class

        Parameters
        ----------
        name: str
            Name given to the test
        pipeline: LLMPipeline
            The pipeline used to generate output
        metrics: list[SingleResultMetric]
            A list of metrics used to compare the pipeline output with the expected output
        """
        super().__init__(name, pipeline, metrics)

    def run_pipeline(self, input_data) -> str:
        """
        Runs the provided pipeline on the input_data

        Parameters
        ----------
        input_data
            The data used for input to the pipeline

        Returns
        -------
        str
            The reply from the pipeline
        """
        return super().run_pipeline(input_data)

    def evaluate(self, input_data, expected_output) -> Dict:
        """
        Evaluates the attached pipeline's output against the expected output using the metrics

        Parameters
        ----------
        input_data
            The data used for input to the pipeline
        expected_output
            The expected result of running the input data through the pipeline

        Returns
        -------
        Dict
            A dictionary of results from evaluating the pipeline.
        """
        return super().evaluate(input_data, expected_output)

    def drop_pipeline(self) -> None:
        self.pipeline.drop()


class EmbeddingComparisonTest(SingleResultPipelineTest):

    def __init__(self, name: str, pipeline, metrics):
        super().__init__(name, pipeline, metrics)

    def run_pipeline(self, input_data):
        return super().run_pipeline(input_data)

    def evaluate(self, input_data, expected_output):
        return super().evaluate(input_data, expected_output)


class RAGPipelineTest(SingleResultPipelineTest):
    """
    This class provides a pipeline test for LLM pipelines that return a single result
    """

    def __init__(
        self,
        name: str,
        pipeline: RAGPipeline,
        metrics: list[SingleResultMetric],
    ):
        """
        Initialises the RAGPipelineTest class

        Parameters
        ----------
        name: str
            Name given to the test
        pipeline: RAGPipeline
            The pipeline used to generate output
        metrics: list[SingleResultMetric]
            A list of metrics used to compare the pipeline output with the expected output
        """
        super().__init__(name, pipeline, metrics)

    def run_pipeline(self, input_data) -> str:
        """
        Runs the provided pipeline on the input_data

        Parameters
        ----------
        input_data
            The data used for input to the pipeline

        Returns
        -------
        str
            The reply from the pipeline
        """
        return super().run_pipeline(input_data)

    def evaluate(self, input_data, expected_output) -> Dict:
        """
        Evaluates the attached pipeline's output against the expected output using the metrics

        Parameters
        ----------
        input_data
            The data used for input to the pipeline
        expected_output
            The expected result of running the input data through the pipeline

        Returns
        -------
        Dict
            A dictionary of results from evaluating the pipeline.
        """
        return super().evaluate(input_data, expected_output)

    def drop_pipeline(self) -> None:
        self.pipeline.drop()
