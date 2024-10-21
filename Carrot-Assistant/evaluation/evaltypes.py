from abc import ABC, abstractmethod
import time
from typing import TypeVar, Generic, Any, List
import json
import os


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
    def run(self, *args, **kwargs) -> Any:
        """
        Run the pipeline
        """
        ...

    def drop(self) -> None:
        pass


M = TypeVar("M", bound=Metric)
P = TypeVar("P", bound=TestPipeline)


class PipelineTest(Generic[P, M]):
    """
    Base class for Pipeline tests
    """

    def __init__(self, name: str, pipeline: P, metrics: list[M]):
        self.name = name
        self.pipeline = pipeline
        self.metrics = metrics

    @abstractmethod
    def run_pipeline(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> dict[str, float]:
        pass

    def drop_pipeline(self) -> None:
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


class SingleResultPipelineTest(PipelineTest[SingleResultPipeline, SingleResultMetric]):
    def __init__(
        self,
        name: str,
        pipeline: SingleResultPipeline,
        metrics: list[SingleResultMetric],
    ):
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
        Evaluate the pipeline by running it on the input data and comparing the result
        to the expected output using all metrics.

        Args:
        input_data: The input data for the pipeline.
        expected_output: The expected output to compare against.

        Returns:
        A dictionary mapping metric names to their calculated values.
        """
        pipeline_output = self.run_pipeline(input_data)
        return {
            metric.__class__.__name__: metric.calculate(
                pipeline_output, expected_output
            )
            for metric in self.metrics
        }


class EvalDataLoader(ABC):
    def __init__(self, file_path) -> None:
        self.file_path = file_path

    @property
    @abstractmethod
    def input_data(self) -> Any:
        pass

    @property
    @abstractmethod
    def expected_output(self) -> Any:
        pass


class EvaluationFramework:
    def __init__(
        self,
        name: str,
        pipeline_tests: List[PipelineTest],
        dataset: EvalDataLoader,
        description: str,
        results_path: str = "results.json",
    ):
        self.name = name
        self._pipeline_tests = pipeline_tests
        self._description = description
        self._results_path = results_path
        self.input_data = dataset.input_data
        self.expected_output = dataset.expected_output

    def run_evaluations(self):
        # Run some tests
        self.evaluation_results = []

        for pipeline_test in self._pipeline_tests:
            result = [
                pipeline_test.evaluate(i, o)
                for i, o in zip(self.input_data, self.expected_output)
            ]
            self.evaluation_results.append({pipeline_test.name: result})

        self._save_evaluations()

    def _save_evaluations(self):
        # Append to 'results.json'
        new_data = {
            "Experiment": self.name,
            "Description": self._description,
            "Time": time.time(),
            "Results": self.evaluation_results,
        }

        if os.path.exists(self._results_path):
            with open(self._results_path, "r") as f:
                previous_runs = json.load(f)
            previous_runs.append(new_data)
        else:
            previous_runs = [new_data]

        with open(self._results_path, "w") as f:
            json.dump(previous_runs, f)
