from evaluation.evaltypes import SingleResultMetric, InformationRetrievalMetric
from rapidfuzz import fuzz


# ------ Single Result Metrics ------
class ExactMatch(SingleResultMetric):
    """
    A metric checking whether the predicted and desired output match.
    This doesn't care what the inputs are.
    """

    def __init__(self) -> None:
        self._description = (
            "Exact match: is the predicted response the same as the ground truth"
        )

    def calculate(self, predicted, actual):
        """
        Calculate the exact match metric.

        Parameters
        ----------
        predicted
            The predicted output from the pipeline.
        actual
            The desired output.

        Returns
        -------
        1 if the predicted and actual outputs match exactly, 0 otherwise.
        """
        return int(predicted == actual)

    @property
    def description(self):
        return self._description


class UncasedMatch(SingleResultMetric):
    """
    A metric for testing whether the predicted and desired outputs are matching strings.
    Case-insensitive and strips whitespace.
    """

    def __init__(self) -> None:
        self._description = "Uncased match: is the predicted response the same as the ground truth, ignoring character case"

    def calculate(self, predicted: str, actual: str) -> float:
        """
        Calculate the exact match metric, if the input value has been wrapped in a list

        Parameters
        ----------
        predicted
            The predicted output from the pipeline
        actual: list
            A list where an item is the desired output of the pipeline
        Returns
        -------
        1 if the predicted and actual outputs match exactly, 0 otherwise
        """
        # We have to do this because the input has to be wrapped in a list for compatibility with prompt templates
        return float(predicted.lower().strip() == actual.lower().strip())

    @property
    def description(self) -> str:
        return self._description


class FuzzyMatchRatio(SingleResultMetric):
    """
    A metric that compares predicted strings to desired output.

    Scores are normalised InDel distance
    """

    def __init__(self) -> None:
        self._description = "Fuzzy Match: the normalized indel similarity between predicted and expected output"

    def calculate(self, predicted: str, actual: str) -> float:
        """
        Calculates the Fuzzy Match Ratio metric

        Parameters
        ----------
        predicted: str
            String output from a SingleResultPipeline
        actual: str
            Ground truth, the string the pipeline is trying to predict
        """
        return fuzz.ratio(predicted.lower(), actual.lower())

    @property
    def description(self):
        return self._description


# -------- Information Retrieval Metrics --------
