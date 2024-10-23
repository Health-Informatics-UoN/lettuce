from evaluation.evaltypes import SingleResultMetric


class ExactMatch(SingleResultMetric):
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
        return float(predicted.lower() == actual.lower())

    @property
    def description(self) -> str:
        return self._description
