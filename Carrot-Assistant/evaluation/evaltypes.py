from abc import ABC, abstractmethod
from typing import TypeVar, Generic

class EvaluationFramework:
    def __init__(self, results_file='results.json'):
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

M = TypeVar('M', bound=Metric)

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

class SingleResultPipelineTest(PipelineTest[SingleResultMetric]):
    def __init__(self, name: str, pipeline: SingleResultPipeline, metrics: list[SingleResultMetric]):
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
        return {metric.__class__.__name__: metric.calculate(pipeline_output, expected_output) 
                for metric in self.metrics}
