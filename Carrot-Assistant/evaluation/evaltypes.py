from abc import ABC, abstractmethod
from typing import TypeVar, Generic


class EvaluationFramework:
    def __init__(self, results_file="results.json"):
        self.results_file = results_file

    def run_evaluations(self):
        # Run some tests
        self._save_evaluations

    def _save_evaluations(self):
        # Append to 'results.json'
        pass


class Metric(ABC):
    """Base class for all metrics."""

    @abstractmethod
    def calculate(self, *args, **kwargs) -> float:
        """
        Calculate the metric value.
        """
        pass


class TestPipeline(ABC):
    """
    Base class for Pipeline runs
    """

    @abstractmethod
    def run(self, *args, **kwargs):
        """
        Run the pipeline
        """
        pass


M = TypeVar("M", bound=Metric)


class PipelineTest(Generic[M]):
    """
    Base class for Pipeline tests
    """

    def __init__(self, name: str, pipeline: TestPipeline, metrics: list[M]):
        self.pipeline = pipeline
        self.metrics = metrics

    @abstractmethod
    def run_pipeline(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> dict[str, float]:
        pass


class SingleResultMetric(Metric):
    """Metric for evaluating pipelines that return a single result."""


class InformationRetrievalMetric(Metric):
    """Metric for evaluating information retrieval pipelines."""

    pass


class SingleResultPipeline(TestPipeline):
    """
    Base class for pipelines returning a single result
    """


class InformationRetrievalPipeline(TestPipeline):
    """
    Base class for pipelines returning multiple results
    """


# ----- Base Class for Single Result Pipeline Tests ----->


class SingleResultPipelineTest(PipelineTest[SingleResultMetric]):
    """
    Base class for testing pipelines that return a single
    result. This class is designed to handle the evaluation
    of pipelines that produce a single result (i.e., a single
    output) for each input provided.
    """

    def __init__(
        self,
        name: str,
        pipeline: SingleResultPipeline,
        metrics: list[SingleResultMetric],
    ):
        """
        Initialize the SingleResultPipelineTest with the given name,
        pipeline, and metrics.

        Parameters
        ----------
        name : str
            The name of the test.

        pipeline : SingleResultPipeline
            The pipeline that will be tested. It is expected to
            return a single result.

        metrics : list[SingleResultMetric]
            A list of metrics (such as ExactMatchMetric) that will
            be used to evaluate the pipeline output.
        """
        super().__init__(name, pipeline, metrics)

    def run_pipeline(self, input_data):
        """
        Run the pipeline with the given input data.

        Args:
        input_data: The input data for the pipeline.

        Returns:
        The result of running the pipeline on the input data.
        """
        return self.pipeline.run(input_data)

    def evaluate(self, input_data, expected_output):
        """
        Evaluate the pipeline by running it on the input data
        and comparing the result to the expected output using all metrics.

        Parameters
        ----------
        input_data
            The input data for the pipeline.

        expected_output
            The expected output to compare against.

        Returns
        -------
        A dictionary mapping metric names to their
        calculated values.
        """
        pipeline_output = self.run_pipeline(input_data)
        return {
            metric.__class__.__name__: metric.calculate(
                pipeline_output, expected_output
            )
            for metric in self.metrics
        }


# ----- Base Class for Information Retrieval or Multiple Concepts Pipeline Tests ----->


class InformationRetrievalPipelineTest(PipelineTest[InformationRetrievalMetric]):
    """
    Base class for testing pipelines that return multiple results.
    This class is designed to evaluate information retrieval metrics
    (e.g., precision, recall) for pipelines returning lists of results.
    """

    def __init__(
        self,
        name: str,
        pipeline: TestPipeline,
        metrics: list[InformationRetrievalMetric],
    ):
        """
        Initialize the InformationRetrievalPipelineTest with
        the given name, pipeline, and metrics.

        Parameters
        ----------
        name : str
            The name of the test.

        pipeline : TestPipeline
            The pipeline that will be tested. It is expected to
            return multiple results.

        metrics : list[InformationRetrievalMetric]
            A list of metrics (such as PrecisionMetric) that will be
            used to evaluate the pipeline output.
        """
        super().__init__(name, pipeline, metrics)

    def run_pipeline(self, input_data):
        """
        Run the pipeline with the given input data and return the list of results.

        Parameters
        ----------
        input_data : Any
            The input data to be processed by the pipeline.

        Returns
        -------
        List
            A list of results generated by the pipeline.
        """
        return self.pipeline.run(input_data)

    def evaluate(self, input_data, expected_output):
        """
        Evaluate the pipeline's multiple results by comparing them to the expected output
        using all the metrics defined.

        Parameters
        ----------
        input_data : Any
            The input data for the pipeline.
        expected_output : List
            The expected output to compare against.

        Returns
        -------
        dict
            A dictionary mapping metric names to their calculated values.
        """
        pipeline_output = self.run_pipeline(input_data)

        # Assert that the pipeline returns more than one result
        assert len(pipeline_output) > 1, (
            "Expected multiple results, but got a single result."
            "Ensure this test is used with pipelines that return multiple results."
            "Consider using SingleResultPipelineTest for pipelines that return a single result."
        )

        # Calculate and return metrics for multiple results
        return {
            metric.__class__.__name__: metric.calculate(
                pipeline_output, expected_output
            )
            for metric in self.metrics
        }
