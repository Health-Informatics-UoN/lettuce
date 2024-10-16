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
    that outputs a list of relationships and concepts or
    simple strings.

    Formula for precision
    ---------------------
    Precision = (number of correct words) / (number of guesses).
    """

    def calculate(self, predicted, actual):
        """
        Calculate precision percentage for each concept or
        single result and return the mean precision.

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
        # If predicted or actual is a list of tuples, assume CoConnect-style data
        if isinstance(predicted[0], (tuple, list)) and isinstance(predicted[0][0], str):
            print("Using CoConnect-style precision calculation")
            # Process CoConnect-style tuples (multiple results)
            return self._calculate_coconnect_precision(predicted, actual)
        else:
            # Process single result predictions (simple string or single concept)
            print("Using single result precision calculation")
            return self._calculate_single_result_precision(predicted, actual)

    def _calculate_coconnect_precision(self, predicted, actual):
        """
        Calculate precision for CoConnect-style data (relationship and concept pairs).

        Parameters
        ----------
        predicted : list of tuples
            The predicted relationships and concepts.

        actual : list of tuples
            The actual (expected) relationships and concepts.

        Returns
        -------
        float
            The mean precision percentage for the CoConnect-style data.
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

    def _calculate_single_result_precision(self, predicted, actual):
        """
        Calculate precision for single result or string-based predictions.

        Parameters
        ----------
        predicted : list of str
            The predicted output strings or single concept (can be multiple words).

        actual : list of str
            The actual (expected) output strings or single concept (can be multiple words).

        Returns
        -------
        float
            The mean precision percentage for the single result.
        """
        total_precision = 0.0
        count = 0

        # Convert the predicted and actual strings into lists of one element if they are not already lists
        if isinstance(predicted, str):
            predicted = [predicted]
        if isinstance(actual, str):
            actual = [actual]

            # Iterate through the predicted and actual values
        for pred_str, actual_str in zip(predicted, actual):

            predicted_words = pred_str.split(" ")
            actual_words = actual_str.split(" ")

            # Calculate the number of matched words
            matched_words = sum(
                1 for pred_word in predicted_words if pred_word in actual_words
            )

            # Precision is the ratio of matched words to the total number of predicted words
            word_precision = (
                matched_words / len(predicted_words)
                if len(predicted_words) > 0
                else 0.0
            )

            total_precision += word_precision * 100  # Convert to percentage
            count += 1

        # Return the mean precision percentage across all predictions
        return total_precision / count if count > 0 else 0.0

    def _word_match_precision(self, predicted_str, actual_str):
        """
        This method calculates precision as the fraction of matched words
        between predicted and actual strings.

        Parameters
        ----------
        predicted_str
            The predicted string.

        actual_str
            The actual string.

        Returns
        -------
        float
            Word precision score.
        """

        predicted_words = set(predicted_str.split())
        actual_words = set(actual_str.split())

        # Find the intersection of predicted and actual words
        matched_words = predicted_words.intersection(actual_words)

        # Precision is the fraction of matched words over the total words in the predicted string
        return (
            len(matched_words) / len(predicted_words)
            if len(predicted_words) > 0
            else 0.0
        )
