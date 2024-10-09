import pytest
from evaluation.evaltypes import (
    SingleResultPipeline,
    InformationRetrievalPipeline,
    SingleResultPipelineTest,
    InformationRetrievalPipelineTest,
)
from evaluation.metrics import ExactMatchMetric, PrecisionMetric


class IdentityPipeline(SingleResultPipeline):
    def run(self, input_data):
        """
        Run the pipeline with the given input data.

        Parameters
        ----------
        input_data
            The input data to be processed by the pipeline.
        """
        return input_data


class ExactMatchTest(SingleResultPipelineTest):
    def __init__(self, name: str, pipeline: SingleResultPipeline):
        """
        Initialize the ExactMatchTest with a name and pipeline.

        Parameters
        ----------
        name
            The name of the test.
        pipeline
            The pipeline to be tested.
        """
        super().__init__(name, pipeline, [ExactMatchMetric()])

    def run_pipeline(self, input_data):
        """
        Run the pipeline and return the result.

        Parameters
        ----------
        input_data
            The input data to be processed by the pipeline.

        Returns
        -------
        The output from the pipeline.
        """
        return self.pipeline.run(input_data)


class PrecisionTest(InformationRetrievalPipelineTest):
    def __init__(self, name: str, pipeline: InformationRetrievalPipeline):
        """
        Initialize the PrecisionTest with a name and pipeline.

        Parameters
        ----------
        name
            The name of the test.
        pipeline
            The pipeline to be tested.
        """
        super().__init__(name, pipeline, [PrecisionMetric()])

    def run_pipeline(self, input_data):
        """
        Run the pipeline and return the result.

        Parameters
        ----------
        input_data
            The input data to be processed by the pipeline.

        Returns
        -------
        The output from the pipeline.
        """
        return self.pipeline.run(input_data)


# ------- Exact Match Tests ------- >


class TestExactMatch:
    @pytest.fixture
    def identity_pipeline(self):
        """
        Fixture to return an instance of the IdentityPipeline.

        Returns
        -------
        IdentityPipeline
            A pipeline that returns the input data unchanged.
        """
        return IdentityPipeline()

    @pytest.fixture
    def exact_match_test(self, identity_pipeline):
        """
        Fixture to create a SingleResultPipelineTest using
        ExactMatchMetric.

        Parameters
        ----------
        identity_pipeline
            The pipeline that will be tested for exact matches.

        Returns
        -------
        SingleResultPipelineTest
            A test that evaluates exact matches between the
            pipeline output and the expected output.
        """
        return SingleResultPipelineTest(
            "Exact Match Test", identity_pipeline, [ExactMatchMetric()]
        )

    @pytest.fixture
    def all_match_dataset(self):
        """
        Fixture to provide a dataset where all input
        and expected output values match exactly.

        Returns
        -------
        list of tuples
            A list of input-output pairs where all inputs
            match the expected outputs exactly.
        """
        return [("input1", "input1"), ("input2", "input2"), ("input3", "input3")]

    @pytest.fixture
    def no_match_dataset(self):
        """
        Fixture to provide a dataset where none
        of the input and expected output values match.

        Returns
        -------
        list of tuples
            A list of input-output pairs where none of
            the inputs match the expected outputs.
        """
        return [("input1", "output1"), ("input2", "output2"), ("input3", "output3")]

    @pytest.fixture
    def half_match_dataset(self):
        """
        Fixture to provide a dataset where some input-output
        pairs match and some do not.

        Returns
        -------
        list of tuples
            A list of input-output pairs where half of the
            inputs match the expected outputs.
        """
        return [
            ("input1", "input1"),
            ("input2", "output2"),
            ("input3", "input3"),
            ("input4", "output4"),
        ]

    def run_test(self, test, dataset):
        """
        Runs the pipeline test on the dataset and
        calculates the average ExactMatchMetric.

        Parameters
        ----------
        test
            The pipeline test to be run.
        dataset
            The dataset of input-output pairs to evaluate.

        Returns
        -------
        float
            The average exact match score across the dataset.
        """
        results = [
            test.evaluate(input_data, expected_output)
            for input_data, expected_output in dataset
        ]
        exact_match_results = [result["ExactMatchMetric"] for result in results]
        return sum(exact_match_results) / len(exact_match_results)

    def test_all_match(self, exact_match_test, all_match_dataset):
        """
        Test to ensure that the exact match test passes
        when all inputs match the expected outputs.

        Parameters
        ----------
        exact_match_test
            The exact match test to be run.
        all_match_dataset
            The dataset where all inputs match the expected outputs.

        Asserts
        -------
        The average exact match score should be 1.0 (100% match).
        """
        assert self.run_test(exact_match_test, all_match_dataset) == 1.0

    def test_no_match(self, exact_match_test, no_match_dataset):
        """
        Test to ensure that the exact match test fails when none of the inputs match the expected outputs.

        Parameters
        ----------
        exact_match_test
            The exact match test to be run.
        no_match_dataset
            The dataset where none of the inputs match the expected outputs.

        Asserts
        -------
        The average exact match score should be 0.0 (0% match).
        """
        assert self.run_test(exact_match_test, no_match_dataset) == 0.0

    def test_half_match(self, exact_match_test, half_match_dataset):
        """
        Test to ensure that the exact match test produces the correct result when only half the inputs match.

        Parameters
        ----------
        exact_match_test
            The exact match test to be run.

        half_match_dataset
            The dataset where half of the inputs match the expected outputs.

        Asserts
        -------
        The average exact match score should be 0.5 (50% match).
        """
        assert self.run_test(exact_match_test, half_match_dataset) == 0.5


# ------- Precision Tests ------- >


class TestPrecisionOnly:
    @pytest.fixture
    def identity_pipeline(self):
        """
        Fixture to return an instance of the IdentityPipeline.

        Returns
        -------
        IdentityPipeline
            A pipeline that returns the input data unchanged.
        """
        return IdentityPipeline()

    @pytest.fixture
    def precision_test(self, identity_pipeline):
        """
        Fixture to create a SingleResultPipelineTest
        using PrecisionMetric.

        Parameters
        ----------
        identity_pipeline
            The pipeline that will be tested for precision.

        Returns
        -------
        SingleResultPipelineTest
            A test that evaluates the precision between the
            pipeline output and the expected output.
        """
        return InformationRetrievalPipelineTest(
            "Precision Test", identity_pipeline, [PrecisionMetric()]
        )

    @pytest.fixture
    def all_match_dataset(self):
        """
        Fixture to provide a dataset where all input and
        expected output values match exactly.

        Returns
        -------
        list of tuples
            A list of input-output pairs where all inputs match
            the expected outputs exactly.
        """
        return [
            (
                [
                    (":relationship", "Maps to", ":concept", "History of event"),
                    (
                        ":relationship",
                        "Maps to value",
                        ":concept",
                        "Malignant neoplasm of skin",
                    ),
                ],
                [
                    (":relationship", "Maps to", ":concept", "History of event"),
                    (
                        ":relationship",
                        "Maps to value",
                        ":concept",
                        "Malignant neoplasm of skin",
                    ),
                ],
            ),
            (
                [
                    (":relationship", "Maps to", ":concept", "History of event"),
                    (":relationship", "Maps to value", ":concept", "Fibromyalgia"),
                ],
                [
                    (":relationship", "Maps to", ":concept", "History of event"),
                    (":relationship", "Maps to value", ":concept", "Fibromyalgia"),
                ],
            ),
        ]

    @pytest.fixture
    def partial_match_dataset(self):
        """
        Fixture to provide a dataset where some input-output
        pairs partially match.

        Returns
        -------
        list of tuples
            A list of input-output pairs where some inputs match the
            expected outputs, while others only partially match.
        """
        return [
            (
                [
                    (":relationship", "Maps to", ":concept", "History of event"),
                    (
                        ":relationship",
                        "Maps to value",
                        ":concept",
                        "Malignant neoplasm of skin",
                    ),
                ],
                [
                    (":relationship", "Maps to", ":concept", "History of event"),
                    (":relationship", "Maps to value", ":concept", "History of event"),
                ],
            ),
            (
                [
                    (":relationship", "Maps to", ":concept", "History of event"),
                    (":relationship", "Maps to value", ":concept", "Fibromyalgia"),
                ],
                [
                    (":relationship", "Maps to", ":concept", "History of event"),
                    (":relationship", "Maps to value", ":concept", "Fibromyalgia"),
                ],
            ),
        ]

    def run_precision_test(self, test, dataset):
        """
        Runs the precision test on the dataset and calculates the average precision.

        Parameters
        ----------
        test
            The precision test to be run.

        dataset
            The dataset of input-output pairs to evaluate.

        Returns
        -------
            The mean precision score across the dataset.
        """
        results = [
            test.evaluate(input_data, expected_output)
            for input_data, expected_output in dataset
        ]
        precision_results = [result["PrecisionMetric"] for result in results]

        for i, (input_data, expected_output) in enumerate(dataset):
            print("\n---------------------")
            print(
                f"Input: {input_data}, Expected: {expected_output}, Precision: {precision_results[i]}"
            )
            print("---------------------\n")

        # Print the overall mean precision
        mean_precision = sum(precision_results) / len(precision_results)
        print("\n---------------------")
        print(f"Mean Precision: {mean_precision}")
        print("---------------------\n")

        return mean_precision

    def test_partial_match(self, precision_test, partial_match_dataset):
        """
        Test to ensure that the precision test handles
        partial matches correctly.

        Parameters
        ----------
        precision_test
            The precision test to be run.

        partial_match_dataset
            The dataset where some inputs match the expected
            outputs and some partially match.

        Asserts
        -------
        The mean precision score should be greater than 0, indicating some matches.
        """

        # Calculate precision for partial match dataset

        print("Running Partial Match Test:")
        actual_precision = self.run_precision_test(
            precision_test, partial_match_dataset
        )
        print(f"Calculated Partial Match Precision: {actual_precision}%")
        assert actual_precision > 0

    def test_all_match(self, precision_test, all_match_dataset):
        """
        Test to ensure that the precision test passes when
        all inputs match the expected outputs.

        Parameters
        ----------
        precision_test
            The precision test to be run.

        all_match_dataset
            The dataset where all inputs match the expected outputs exactly.

        Asserts
        -------
        The mean precision score should be 100.0%.
        """
        # Calculate precision for all match dataset
        print("Running All Match Test:")
        actual_precision = self.run_precision_test(precision_test, all_match_dataset)
        print(f"Calculated All Match Precision: {actual_precision}%")
        assert actual_precision == 100.0


# pytest -s test_evals.py (To run all the tests)
