from evaluation.evaltypes import SingleResultMetric, InformationRetrievalMetric


class ExactMatchMetric(SingleResultMetric):
    def calculate(self, predicted, actual):
        """
        This class calculate the exact match metric.

        Parameters
        ----------
        predicted
            The predicted output from the pipeline.

        actual
            The desired output.

        Returns
        -------
        the exact match metric
        """
        return int(predicted == actual)


class PrecisionMetric(InformationRetrievalMetric):
    """
    This class calculates the precision metric for a pipeline
    that outputs a list of relationships and concepts.
    """

    def calculate(self, predicted, actual):
        """
        Calculate precision percentage for each concept and return the mean precision.

        Parameters
        ----------
        predicted
            The predicted output from the pipeline.

        actual
            The desired output.

        Returns
        -------
        Mean precision percentage.
        """
        predicted_relationships, actual_relationships = predicted, actual

        total_precision = 0.0
        count = 0

        # Compare each pair of (relationship, concept) from predicted and actual
        for (pred_rel, pred_rel_value, pred_concept, pred_concept_value), (
            act_rel,
            act_rel_value,
            act_concept,
            act_concept_value,
        ) in zip(predicted_relationships, actual_relationships):

            # Calculate word precision for the relationship and concept
            rel_precision = self._word_match_precision(pred_rel_value, act_rel_value)
            concept_precision = self._word_match_precision(
                pred_concept_value, act_concept_value
            )

            # Calculate the average precision for this concept (relationship + concept)
            avg_precision = (rel_precision + concept_precision) / 2

            # Convert to percentage and add to total
            total_precision += avg_precision * 100
            count += 1

        # Return the mean precision percentage
        return total_precision / count if count > 0 else 0.0

    def _word_match_precision(self, predicted_str, actual_str):
        """
        This method calculate precision as the fraction of matched words
        between predicted and actual strings.

        Parameters
        ----------
        predicted_str
            The predicted string.

        actual_str
            The actual string.
        """
        predicted_words = set(predicted_str.split())
        actual_words = set(actual_str.split())

        # Find the intersection of predicted and actual words
        matched_words = predicted_words.intersection(actual_words)

        # Precision is the fraction of matched words over the total words in the actual string
        return len(matched_words) / len(actual_words) if len(actual_words) > 0 else 0.0
