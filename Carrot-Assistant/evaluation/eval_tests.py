from evaluation.evaltypes import (
    SingleResultPipelineTest,
    SingleResultMetric,
)
from evaluation.pipelines import LLMPipeline


class LLMPipelineTest(SingleResultPipelineTest):
    def __init__(
        self,
        name: str,
        pipeline: LLMPipeline,
        metrics: list[SingleResultMetric],
    ):
        super().__init__(name, pipeline, metrics)

    def run_pipeline(self, input_data):
        return super().run_pipeline(input_data)

    def evaluate(self, input_data, expected_output):
        return super().evaluate(input_data, expected_output)
