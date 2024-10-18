import pytest
from jinja2 import Environment, Template

from evaluation.evaltypes import SingleResultPipeline, SingleResultPipelineTest
from evaluation.metrics import ExactMatchMetric
from evaluation.pipelines import LLMPipeline

from options.pipeline_options import LLMModel


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
        return SingleResultPipelineTest(
            "Exact Match Test", identity_pipeline, [ExactMatchMetric()]
        )

    @pytest.fixture
    def all_match_dataset(self):
        return [("input1", "input1"), ("input2", "input2"), ("input3", "input3")]

    @pytest.fixture
    def no_match_dataset(self):
        return [("input1", "output1"), ("input2", "output2"), ("input3", "output3")]

    @pytest.fixture
    def half_match_dataset(self):
        return [
            ("input1", "input1"),
            ("input2", "output2"),
            ("input3", "input3"),
            ("input4", "output4"),
        ]

    def run_test(self, test, dataset):
        results = [
            test.evaluate(input_data, expected_output)
            for input_data, expected_output in dataset
        ]
        exact_match_results = [result["ExactMatchMetric"] for result in results]
        return sum(exact_match_results) / len(exact_match_results)

    def test_all_match(self, exact_match_test, all_match_dataset):
        assert self.run_test(exact_match_test, all_match_dataset) == 1.0

    def test_no_match(self, exact_match_test, no_match_dataset):
        assert self.run_test(exact_match_test, no_match_dataset) == 0.0

    def test_half_match(self, exact_match_test, half_match_dataset):
        assert self.run_test(exact_match_test, half_match_dataset) == 0.5


# LLM pipeline tests


class TestBasicLLM:
    @pytest.fixture
    def llm_prompt(self):
        env = Environment()
        template = env.from_string(
            """
                                   You are a parrot that repeats whatever is said to you, with no explanation. You will be given a sentence as input, repeat it.

                                   Sentence: {{input_sentence}}
                                   """
        )
        return template

    @pytest.fixture
    def llm_pipeline(self, llm_prompt):
        return LLMPipeline(LLMModel["llama-3.1-8b"], llm_prompt)

    def test_returns_string(self, llm_pipeline):
        model_output = llm_pipeline.run({"input_sentence": "Polly wants a cracker"})
        assert isinstance(model_output, str)
