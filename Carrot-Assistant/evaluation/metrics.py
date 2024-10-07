from evaluation.evaltypes import SingleResultMetric

class ExactMatchMetric(SingleResultMetric):
    def calculate(self, predicted, actual):
        """
        Calculate the exact match metric.
        
        Args:
        predicted: The predicted output from the pipeline.
        actual: The desired output.
        
        Returns:
        1 if the predicted and actual outputs match exactly, 0 otherwise.
        """
        return int(predicted == actual)
