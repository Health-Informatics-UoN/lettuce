import pytest
from evaluation.evaltypes import SingleResultPipeline, SingleResultPipelineTest
from evaluation.metrics import ExactMatchMetric

class IdentityPipeline(SingleResultPipeline):
    def run(self, input_data):
        return input_data

class ExactMatchTest(SingleResultPipelineTest):
    def __init__(self, name: str, pipeline: SingleResultPipeline):
        super().__init__(name, pipeline, [ExactMatchMetric()])
    
    def run_pipeline(self, input_data):
        return self.pipeline.run(input_data)
    
class TestExactMatch:
    @pytest.fixture
    def identity_pipeline(self):
        return IdentityPipeline()

    @pytest.fixture
    def exact_match_test(self, identity_pipeline):
        return SingleResultPipelineTest("Exact Match Test", identity_pipeline, [ExactMatchMetric()])

    @pytest.fixture
    def all_match_dataset(self):
        return [("input1", "input1"), ("input2", "input2"), ("input3", "input3")]

    @pytest.fixture
    def no_match_dataset(self):
        return [("input1", "output1"), ("input2", "output2"), ("input3", "output3")]

    @pytest.fixture
    def half_match_dataset(self):
        return [("input1", "input1"), ("input2", "output2"), ("input3", "input3"), ("input4", "output4")]

    def run_test(self, test, dataset):
        results = [test.evaluate(input_data, expected_output) for input_data, expected_output in dataset]
        exact_match_results = [result['ExactMatchMetric'] for result in results]
        return sum(exact_match_results) / len(exact_match_results)    

    def test_all_match(self, exact_match_test, all_match_dataset):
        assert self.run_test(exact_match_test, all_match_dataset) == 1.0

    def test_no_match(self, exact_match_test, no_match_dataset):
        assert self.run_test(exact_match_test, no_match_dataset) == 0.0

    def test_half_match(self, exact_match_test, half_match_dataset):
        assert self.run_test(exact_match_test, half_match_dataset) == 0.5
