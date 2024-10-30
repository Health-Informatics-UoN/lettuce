import pytest
from jinja2 import Environment
import os
import json

from evaluation.eval_tests import LLMPipelineTest
from evaluation.evaltypes import (
    SingleResultPipeline,
    SingleResultPipelineTest,
    EvaluationFramework,
)
from evaluation.metrics import ExactMatch, PrecisionMetric, RecallMetric, FScoreMetric
from evaluation.pipelines import LLMPipeline
from evaluation.eval_data_loaders import SingleInputSimpleCSV

from options.pipeline_options import LLMModel


class IdentityPipeline(SingleResultPipeline):
    def run(self, input_data):
        return input_data


class ExactMatchTest(SingleResultPipelineTest):
    def __init__(self, name: str, pipeline: SingleResultPipeline):
        super().__init__(name, pipeline, [ExactMatch()])

    def run_pipeline(self, input_data):
        return self.pipeline.run(input_data)


class TestExactMatch:
    @pytest.fixture
    def identity_pipeline(self):
        return IdentityPipeline()

    @pytest.fixture
    def exact_match_test(self, identity_pipeline):
        return SingleResultPipelineTest(
            "Exact Match Test", identity_pipeline, [ExactMatch()]
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
        exact_match_results = [result["ExactMatch"] for result in results]
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
        return LLMPipeline(LLMModel.LLAMA_3_1_8B, llm_prompt, ["input_sentence"])

    def test_returns_string(self, llm_pipeline):
        model_output = llm_pipeline.run(["Polly wants a cracker"])
        assert isinstance(model_output, str)

    @pytest.fixture
    def llm_pipeline_test(self, llm_pipeline):
        return LLMPipelineTest("Parrot Pipeline", llm_pipeline, [ExactMatch()])

    def test_pipeline_called_from_eval_returns_string(self, llm_pipeline_test):
        model_output = llm_pipeline_test.run_pipeline(["Polly wants a cracker"])
        assert isinstance(model_output, str)

    def test_llm_pipelinetest_evaluates(self, llm_pipeline_test):
        model_eval = llm_pipeline_test.evaluate(
            input_data={"input_sentence": "Polly wants a cracker"},
            expected_output="Polly wants a cracker",
        )
        assert isinstance(model_eval, dict)


class TestEvaluationFramework:
    @pytest.fixture
    def identity_pipeline(self):
        return IdentityPipeline()

    @pytest.fixture
    def exact_match_test(self, identity_pipeline):
        return SingleResultPipelineTest(
            name="Exact Match Test",
            pipeline=identity_pipeline,
            metrics=[ExactMatch()],
        )

    @pytest.fixture(scope="session")
    def matching_set(self, tmp_path_factory):
        tmp_dir = tmp_path_factory.mktemp("data")
        csv_path = tmp_dir / "matching_set.csv"
        filestring = """input_data,expected_output
input1,input1
input2,input2
input3,input3"""
        csv_path.write_text(filestring)
        return str(csv_path)

    @pytest.fixture
    def data_loader(self, matching_set):
        return SingleInputSimpleCSV(matching_set)

    @pytest.fixture
    def eval_framework(self, exact_match_test, data_loader, tmp_path):
        return EvaluationFramework(
            name="Test Experiment",
            pipeline_tests=[exact_match_test],
            dataset=data_loader,
            description="Test Description",
            results_path=str(tmp_path / "results.json"),
        )

    def test_creates_new_results_file(self, eval_framework, tmp_path):
        results_path = tmp_path / "results.json"

        assert not os.path.exists(results_path)

        eval_framework.run_evaluations()

        assert os.path.exists(results_path)

        with open(results_path, "r") as f:
            results = json.load(f)

        assert len(results) == 1
        assert results[0]["Experiment"] == "Test Experiment"

    def test_appends_to_existing_results_file(self, eval_framework, tmp_path):
        results_path = tmp_path / "results.json"

        eval_framework.run_evaluations()

        eval_framework.run_evaluations()

        with open(results_path, "r") as f:
            results = json.load(f)

        assert len(results) == 2
        assert results[0]["Experiment"] == "Test Experiment"
        assert results[1]["Experiment"] == "Test Experiment"


class TestInformationRetrievalMetrics:
    @pytest.fixture
    def text_relevant_instances(self):
        return [
            "Mrs Doubtfire",
            "Good Morning, Vietnam",
            "Patch Adams",
            "Good Will Hunting",
            "Aladdin",
            "Dead Poets Society",
        ]

    @pytest.fixture
    def text_retrieved_instances(self):
        return {
            "text_prediction_1": [
                "Mrs Doubtfire",
                "Good Morning, Vietnam",
                "Patch Adams",
                "Good Will Hunting",
                "Aladdin",
                "Dead Poets Society",
            ],
            "text_prediction_2": [
                "Mrs Doubtfire",
                "Good Morning, Vietnam",
                "Patch Adams",
            ],
            "text_prediction_3": [
                "Mrs Doubtfire",
                "Good Morning, Vietnam",
                "Patch Adams",
                "Good Will Hunting",
                "Aladdin",
                "Dead Poets Society",
                "Walk the Line",
                "O Brother, Where Art Thou?",
                "Gladiator",
                "The Village",
                "Her",
                "Inherent Vice",
            ],
        }

    @pytest.fixture
    def precision_metric(self) -> PrecisionMetric:
        return PrecisionMetric()

    @pytest.fixture
    def recall_metric(self) -> RecallMetric:
        return RecallMetric()

    @pytest.fixture
    def f_1_score(self) -> FScoreMetric:
        return FScoreMetric(1)

    def test_precision(
        self,
        text_relevant_instances: list,
        text_retrieved_instances: dict,
        precision_metric: PrecisionMetric,
    ):
        precision_results = [
            precision_metric.calculate(retrieved, text_relevant_instances)
            for retrieved in text_retrieved_instances.values()
        ]

        assert precision_results == [1.0, 1.0, 0.5]

    def test_recall(
        self,
        text_relevant_instances: list,
        text_retrieved_instances: dict,
        recall_metric: RecallMetric,
    ):
        recall_results = [
            recall_metric.calculate(retrieved, text_relevant_instances)
            for retrieved in text_retrieved_instances.values()
        ]

        assert recall_results == [1, 0.5, 1]

    def test_f_1_score(
        self,
        text_relevant_instances: list,
        text_retrieved_instances: dict,
        f_1_score: FScoreMetric,
    ):
        f_score_results = [
            f_1_score.calculate(retrieved, text_relevant_instances)
            for retrieved in text_retrieved_instances.values()
        ]

        assert f_score_results == [1, 2 / 3, 2 / 3]
