from evaluation.evaltypes import SingleResultMetric, InformationRetrievalMetric

# ----------- Accuracy Metric ----------- >


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


# ----------- Precision Metric ----------- >


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
            # Process CoConnect-style tuples (multiple results)
            return self._calculate_coconnect_precision(predicted, actual)
        else:
            # Process single result predictions (simple string or single concept)
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


# ----------- Recall Metric ----------- >


class RecallMetric(InformationRetrievalMetric):
    """
    This class calculates the recall metric for a pipeline
    that outputs a list of relationships and concepts or
    simple strings.

    Formula for recall
    ---------------------
    Recall = (number of correct words) /
    (number of words in ground truth).
    """

    def calculate(self, predicted, actual):
        """
        Calculate recall percentage for each concept or single
        result and return the mean recall.

        Parameters
        ----------
        predicted
            The predicted output from the pipeline.

        actual
            The desired output.

        Returns
        -------
        Mean recall percentage.
        """
        # If predicted or actual is a list of tuples, assume CoConnect-style data
        if isinstance(predicted[0], (tuple, list)) and isinstance(predicted[0][0], str):

            # Process CoConnect-style tuples (multiple results)
            return self._calculate_coconnect_recall(predicted, actual)
        else:
            # Process single result predictions (simple string or single concept)
            return self._calculate_single_result_recall(predicted, actual)

    def _calculate_coconnect_recall(self, predicted, actual):
        """
        Calculate recall for CoConnect-style data (relationship
        and concept pairs).

        Parameters
        ----------
        predicted : list of tuples
            The predicted relationships and concepts.

        actual : list of tuples
            The actual (expected) relationships and concepts.

        Returns
        -------
        float
            The mean recall percentage for the CoConnect-style data.
        """
        predicted_relationships, actual_relationships = predicted, actual

        total_recall = 0.0
        count = 0

        # Compare each pair of (relationship, concept) from predicted and actual
        for (pred_rel, pred_rel_value, pred_concept, pred_concept_value), (
            act_rel,
            act_rel_value,
            act_concept,
            act_concept_value,
        ) in zip(predicted_relationships, actual_relationships):

            # Calculate word recall for the relationship and concept
            rel_recall = self._word_match_recall(pred_rel_value, act_rel_value)
            concept_recall = self._word_match_recall(
                pred_concept_value, act_concept_value
            )

            # Calculate the average recall for this concept (relationship + concept)
            avg_recall = (rel_recall + concept_recall) / 2

            # Convert to percentage and add to total
            total_recall += avg_recall * 100
            count += 1

        # Return the mean recall percentage
        return total_recall / count if count > 0 else 0.0

    def _calculate_single_result_recall(self, predicted, actual):
        """
        Calculate recall for single result or string-based predictions.

        Parameters
        ----------
        predicted : list of str
            The predicted output strings or single concept
            (can be multiple words).

        actual : list of str
            The actual (expected) output strings or single
            concept (can be multiple words).

        Returns
        -------
        float
            The mean recall percentage for the single result.
        """
        total_recall = 0.0
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
                1 for actual_word in actual_words if actual_word in predicted_words
            )

            # Recall is the ratio of matched words to the total number of actual words
            word_recall = (
                matched_words / len(actual_words) if len(actual_words) > 0 else 0.0
            )

            total_recall += word_recall * 100  # Convert to percentage
            count += 1

        # Return the mean recall percentage across all predictions
        return total_recall / count if count > 0 else 0.0

    def _word_match_recall(self, predicted_str, actual_str):
        """
        This method calculates recall as the fraction of matched
        words between predicted and actual strings.

        Parameters
        ----------
        predicted_str
            The predicted string.

        actual_str
            The actual string.

        Returns
        -------
        float
            Word recall score.
        """
        predicted_words = set(predicted_str.split())
        actual_words = set(actual_str.split())

        # Find the intersection of predicted and actual words
        matched_words = predicted_words.intersection(actual_words)

        # Recall is the fraction of matched words over the total words in the actual string
        return len(matched_words) / len(actual_words) if len(actual_words) > 0 else 0.0


# ----------- F1 Score Metric ----------- >


class F1ScoreMetric(InformationRetrievalMetric):
    """
    This class calculates the F1 score for a pipeline that
    outputs a list  of relationships and concepts or simple strings.

    Formula for F1 Score
    ---------------------
    F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    """

    def calculate(self, predicted, actual):
        """
        Calculate the average F1 score across all instances.

        Parameters
        ----------
        predicted
            The predicted output from the pipeline.

        actual
            The desired output.

        Returns
        -------
        Mean F1 score percentage.
        """
        if not isinstance(predicted[0], (tuple, list)) and isinstance(
            predicted[0][0], str
        ):

            if isinstance(predicted, str):
                predicted = [predicted]

            if isinstance(actual, str):
                actual = [actual]

        precision_metric = PrecisionMetric()
        recall_metric = RecallMetric()

        total_f1_score = 0.0
        count = 0

        # Iterate through the predicted and actual values
        for pred_str, actual_str in zip(predicted, actual):

            # Calculate precision and recall for each instance
            precision = precision_metric.calculate([pred_str], [actual_str])
            recall = recall_metric.calculate([pred_str], [actual_str])

            # Assert that precision and recall calculations were performed
            assert precision is not None, "Precision calculation failed."
            assert recall is not None, "Recall calculation failed."

            # Handle edge case when precision and recall are both 0
            if precision + recall == 0:
                f1_score = 0.0
            else:
                f1_score = 2 * (precision * recall) / (precision + recall)

            total_f1_score += f1_score
            count += 1

            # Print debug information
            print(
                f"Predicted: {pred_str}, Actual: {actual_str}, Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}"
            )

        # Return the mean F1 score across all instances
        return total_f1_score / count if count > 0 else 0.0
