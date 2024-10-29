from typing import Any, List
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
def calc_precision(relevant_instances: List[Any], prediction: List[Any]) -> float:
    """
    Compares two lists and calculates precision

    Precision = (Number of relevant instances retrieved)/(Number of instances retrieved)

    Parameters
    ----------
    relevant_instances: List[Any]
        The set of relevant instances, or positive class
    prediction: List[Any]
        A prediction made by an information retrieval system

    Returns
    -------
    float
        A score for the precision
    """
    relevant_retrieved_instances = [x for x in prediction if x in relevant_instances]
    return len(relevant_retrieved_instances) / len(prediction)


def calc_recall(relevant_instances: List[Any], prediction: List[Any]) -> float:
    """
    Compares two lists and calculates recall

    Recall = (Number of relevant instances retrieved)/(Number of relevant instances)

    Parameters
    ----------
    relevant_instances: List[Any]
        The set of relevant instances, or positive class
    prediction: List[Any]
        A prediction made by an information retrieval system

    Returns
    -------
    float
        A score for the recall
    """
    relevant_retrieved_instances = [x for x in prediction if x in relevant_instances]
    return len(relevant_retrieved_instances) / len(relevant_instances)


class PrecisionMetric(InformationRetrievalMetric):
    def __init__(self) -> None:
        self._description = "Precision: a type agnostic precision metric"

    def calculate(self, predicted: List[Any], actual: List[Any]) -> float:
        """
        Calculates precision for the information retrieval pipeline's prediction against a positive set

        Parameters
        ----------
        predicted: List[Any]
            The output of an information retrieval pipeline
        actual: List[Any]
            The set of relevant instances for the input
        """
        return calc_precision(actual, predicted)

    @property
    def description(self) -> str:
        return self._description


class RecallMetric(InformationRetrievalMetric):
    def __init__(self) -> None:
        self._description = "Recall: a type agnostic recall metric"

    def calculate(self, predicted: List[Any], actual: List[Any]) -> float:
        """
        Calculates recall for the information retrieval pipeline's prediction against a positive set

        Parameters
        ----------
        predicted: List[Any]
            The output of an information retrieval pipeline
        actual: List[Any]
            The set of relevant instances for the input
        """
        return calc_recall(actual, predicted)

    @property
    def description(self) -> str:
        return self._description


class FScoreMetric(InformationRetrievalMetric):
    def __init__(self, beta: float) -> None:
        """
        Initialises the F-Score metric

        Parameters
        ----------
        beta: float
            The ratio by which to weight precision to recall
        """
        self._description = (
            "F Score: a type agnostic F-score metric with a Beta of " + str(beta)
        )
        self._beta = beta

    def calculate(self, predicted: List[Any], actual: List[Any]) -> float:
        """
        Calculates F score with the class beta for the information retrieval pipeline's prediction against a positive set

        Parameters
        ----------
        predicted: List[Any]
            The output of an information retrieval pipeline
        actual: List[Any]
            The set of relevant instances for the input
        """
        precision = calc_precision(actual, predicted)
        recall = calc_recall(actual, predicted)
        return (
            (1 + self._beta**2)
            * (precision * recall)
            / ((self._beta**2 * precision) + recall)
        )

    @property
    def description(self) -> str:
        return self._description
