import pytest
from jinja2 import Environment
import os
import json
from sqlalchemy.orm import Session

from evaluation.eval_tests import LLMPipelineTest
from evaluation.evaltypes import (
    SingleResultPipeline,
    SingleResultPipelineTest,
    EvaluationFramework,
)
from evaluation.metrics import (
    AncestorNameUncasedMatch,
    ExactMatch,
    FuzzyMatchRatio,
    PrecisionMetric,
    RecallMetric,
    FScoreMetric,
    AncestorNamePrecision,
    RelatedNamePrecision,
    RelatedNameUncasedMatch,
)
from evaluation.pipelines import LLMPipeline
from evaluation.eval_data_loaders import SingleInputSimpleCSV
from omop.db_manager import db_session

from options.pipeline_options import LLMModel


# --- Single result metrics ---
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


# --- Information Retrieval Metrics ---
class TestDatabaseMetrics:
    @pytest.fixture
    def db_connection(self) -> Session:
        return db_session()

    @pytest.fixture
    def pretend_relationship_matches(self) -> dict[str, list[str]]:
        return {
            "full match": [
                "Retired SNOMED UK Drug extension concept, do not use, use concept indicated by the CONCEPT_RELATIONSHIP table, if any",
                "Retired SNOMED UK Drug extension concept, do not use, use concept indicated by the CONCEPT_RELATIONSHIP table, if any",
                "acetaminophen / dextromethorphan Oral Powder Product",
                "acetaminophen, dextromethorphan hbr, phenylephrine hcl, doxylamine succinate KIT [daytime/nighttime cold and flu multi-symptom relief]",
                "acetaminophen / dextromethorphan / phenylephrine / triprolidine Pill",
                "Acetaminophen- and orphenadrine-containing product",
            ],
            "half match": [
                "Retired SNOMED UK Drug extension concept, do not use, use concept indicated by the CONCEPT_RELATIONSHIP table, if any",
                "Retired SNOMED UK Drug extension concept, do not use, use concept indicated by the CONCEPT_RELATIONSHIP table, if any",
                "acetaminophen / dextromethorphan Oral Powder Product",
                "grunge",
                "nato",
                "chumbawumba",
            ],
            "no match": ["banana", "spoon", "eagle", "grunge", "nato", "chumbawumba"],
        }

    def test_related_name_precision(self, db_connection, pretend_relationship_matches):
        metric = RelatedNamePrecision(db_connection, ["RxNorm"])
        full_match = metric.calculate(
            pretend_relationship_matches["full match"], "Acetaminophen"
        )
        half_match = metric.calculate(
            pretend_relationship_matches["half match"], "Acetaminophen"
        )
        no_match = metric.calculate(
            pretend_relationship_matches["no match"], "Acetaminophen"
        )

        assert full_match == 1.0
        assert half_match == 0.5
        assert no_match == 0

    @pytest.fixture
    def pretend_ancestor_matches(self) -> dict[str, list[str]]:
        return {
            "full match": [
                "homatropine methylbromide; systemic",
                "codeine and other non-opioid analgesics; systemic",
                "dihydrocodeine and other non-opioid analgesics; systemic",
                "tramadol and other non-opioid analgesics; systemic",
                "tropenzilone and analgesics",
                "pitofenone and analgesics; systemic",
            ],
            "half match": [
                "homatropine methylbromide; systemic",
                "codeine and other non-opioid analgesics; systemic",
                "dihydrocodeine and other non-opioid analgesics; systemic",
                "ice cream",
                "laserdisc",
                "spoon again",
            ],
            "no match": [
                "ice cream",
                "laserdisc",
                "spoon again",
                "unbelievably, another spoon",
                "ladle",
                "potato",
            ],
        }

    def test_ancestor_name_precision(self, db_connection, pretend_ancestor_matches):
        metric = AncestorNamePrecision(db_connection, ["RxNorm"])
        full_match = metric.calculate(
            pretend_ancestor_matches["full match"], "Acetaminophen"
        )
        half_match = metric.calculate(
            pretend_ancestor_matches["half match"], "Acetaminophen"
        )
        no_match = metric.calculate(
            pretend_ancestor_matches["no match"], "Acetaminophen"
        )

        assert full_match == 1.0
        assert half_match == 0.5
        assert no_match == 0

    def test_related_name_uncased_match(self, db_connection):
        metric = RelatedNameUncasedMatch(db_connection, ["RxNorm"])

        concept_match = metric.calculate("acetaminophen", "Acetaminophen")
        concept_related = metric.calculate(
            "acetaminophen / dextromethorphan Oral Powder Product", "Acetaminophen"
        )
        concept_not_related = metric.calculate("Oxford Circus", "Acetaminophen")

        assert concept_match == 1.0
        assert concept_related == 1.0
        assert concept_not_related == 0

    def test_ancestor_name_uncased_match(self, db_connection):
        metric = AncestorNameUncasedMatch(db_connection, ["RxNorm"])

        concept_match = metric.calculate("acetaminophen", "Acetaminophen")
        concept_related = metric.calculate(
            "codeine and other non-opioid analgesics; systemic", "Acetaminophen"
        )
        concept_not_related = metric.calculate("Oxford Circus", "Acetaminophen")

        assert concept_match == 1.0
        assert concept_related == 1.0
        assert concept_not_related == 0


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
        return LLMPipeline(
            LLMModel.LLAMA_3_1_8B, llm_prompt, template_vars=["input_sentence"]
        )

    def test_returns_string(self, llm_pipeline):
        model_output = llm_pipeline.run({"input_sentence": "Polly wants a cracker"})
        assert isinstance(model_output, str)

    @pytest.fixture
    def llm_pipeline_test(self, llm_pipeline):
        return LLMPipelineTest("Parrot Pipeline", llm_pipeline, [ExactMatch()])

    def test_pipeline_called_from_eval_returns_string(self, llm_pipeline_test):
        model_output = llm_pipeline_test.run_pipeline(
            {"input_sentence": "Polly wants a cracker"}
        )
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
