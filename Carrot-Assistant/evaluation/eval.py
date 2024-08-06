from abc import ABC, abstractmethod

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
    def calculate(self, *args, **kwargs):
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

class PipelineTest(ABC):
    """
    Base class for Pipeline tests
    """
    def __init__(self, name: str, pipeline: TestPipeline, metrics: list[Metric]):
        self.pipeline = pipeline
        self.metrics = metrics

    @abstractmethod
    def run_pipeline(self, *args, **kwargs):
        pass
    @abstractmethod
    def evaluate(self, *args, **kwargs):
    	pass

class SingleResultMetric(Metric):
    """Metric for evaluating pipelines that return a single result."""
    pass

class InformationRetrievalMetric(Metric):
    """Metric for evaluating information retrieval pipelines."""
    pass

class SingleResultPipeline(TestPipeline):
	"""
	Base class for pipelines returning a single result
	"""

class SingleResultPipelineTest(PipelineTest):
    def __init__(self, name: str, pipeline: SingleResultPipeline, metrics: list[SingleResultMetric]):
        self.pipeline = pipeline
        self.metrics = metrics
