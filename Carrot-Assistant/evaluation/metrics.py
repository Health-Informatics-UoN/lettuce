import pandas as pd
from difflib import SequenceMatcher

class Metrics:
    """
    This class is used to calculate the metrics for the evaluation of the model.
    
    Parameters:
    -----------
    df: pd.DataFrame
        The DataFrame containing the test dataset.
        
    logger: Logger
        The logger object to log the results of the evaluation.
    """
    
    def __init__(self, df: pd.DataFrame, logger):
        
        self.df = df
        self.logger = logger
    
    def informal_name_match(self):
        """
        This function calculates the percentage of 
        informal names that have an OMOP match in the dataset.
        
        Parameters:
        -----------
        None
        
        Returns:
        --------
        informal_name_match_results: dict
            A dictionary containing the results of the informal name match.
            
        Raises:
        -------
        ValueError: 
            If the column 'Informal OMOP Match' is not in the DataFrame.
        """
        self.name = 'informal_name_match'
        
        if 'Informal OMOP Match' not in self.df.columns:
            raise ValueError("Column 'Informal OMOP Match' is not in the dataframe.")
        
        try:
            
            # Count the number of informal names that have an OMOP match
            informal_name_count = self.df['Informal OMOP Match'].str.lower().eq('yes').sum()
            no_match_count = self.df['Informal OMOP Match'].str.lower().eq('no').sum()

            total_count = len(self.df)

            # Calculate the percentage of informal names that have an OMOP match
            informal_match_percentage = (informal_name_count / total_count) * 100 if total_count > 0 else 0
            no_match_percentage = (no_match_count / total_count) * 100 if total_count > 0 else 0


            print(f'total_count: {total_count}')
            print(f'informal_name_count: {informal_name_count}')
            print(f'Informal OMOP Match - Yes (%): {informal_match_percentage}')
            print(f'Informal OMOP Match - No (%): {no_match_percentage}')
            print("=" * 50)
            
            self.logger.update_log('informal_name_match', {
                'total': total_count,
                'match_count': informal_name_count,
                'no_match_count': no_match_count,
                'informal_match_percentage': informal_match_percentage,
                'no_match_percentage': no_match_percentage,
            })
            
            return {
                'total': total_count,
                'match_count': informal_name_count,
                'no_match_count': no_match_count,
                'informal_match_percentage': informal_match_percentage,
                'no_match_percentage': no_match_percentage,
            }
            
        except Exception as e:
            raise ValueError(f"Error calculating informal name match: {e}")
        
    
    def llm_output_match(self):
        """
        This function calculates the percentage of LLM output that has an OMOP match.
        
        Parameters:
        -----------
        None
        
        Returns:
        --------
        llm_output_match_results: dict
            A dictionary containing the results of the LLM output match.
            
        Raises:
        -------
        ValueError:
            If the column 'LLM OMOP Match' is not in the DataFrame.
        """
        self.name = 'llm_output_match'
        
        if 'LLM OMOP Match' not in self.df.columns:
            raise ValueError("Column 'LLM OMOP Match' is not in the dataframe.")
        
        try:
            
            # Count the number of LLM output that has an OMOP match
            llm_omop_match = self.df['LLM OMOP Match'].str.lower().eq('yes').sum()
            llm_no_match = self.df['LLM OMOP Match'].str.lower().eq('no').sum()
            llm_not_used = self.df['LLM OMOP Match'].str.lower().eq('not using llm').sum()
            
            total_count = len(self.df)
            
            total_llm_count = llm_omop_match + llm_no_match
            
            # Calculate the percentage of LLM output that has an OMOP match
            llm_omop_match_percentage = (llm_omop_match / total_llm_count) * 100 if total_llm_count > 0 else 0
            llm_no_match_percentage = (llm_no_match / total_llm_count) * 100 if total_llm_count > 0 else 0
            
            llm_not_used_percentage = (llm_not_used / total_count) * 100 if total_count > 0 else 0
            

            print(f'Total Count: {total_count}')
            print(f'Total LLM Count: {total_llm_count}')
            print(f'LLM OMOP Match - Yes: {llm_omop_match}')
            print(f'LLM OMOP Match - No: {llm_no_match}')
            print(f'LLM OMOP Match - Not Using LLM: {llm_not_used}')
            print(f'LLM OMOP Match - Yes (%): {llm_omop_match_percentage}')
            print(f'LLM OMOP Match - No (%): {llm_no_match_percentage}')
            print(f'LLM OMOP Match - Not Using LLM (%): {llm_not_used_percentage}')
            print("=" * 50)
            
            self.logger.update_log('llm_output_match', {
                'total': total_count,
                'total_llm_used': total_llm_count,
                'llm_omop_match': llm_omop_match,
                'llm_no_match': llm_no_match,
                'llm_not_used': llm_not_used,
                'llm_omop_match_percentage': llm_omop_match_percentage,
                'llm_no_match_percentage': llm_no_match_percentage,
                'llm_not_used_percentage': llm_not_used_percentage,
            })
            
            return {
                'total': total_count,
                'total_llm_used': total_llm_count,
                'llm_omop_match': llm_omop_match,
                'llm_no_match': llm_no_match,
                'llm_not_used': llm_not_used,
                'llm_omop_match_percentage': llm_omop_match_percentage,
                'llm_no_match_percentage': llm_no_match_percentage,
                'llm_not_used_percentage': llm_not_used_percentage,
            }
        
        except Exception as e:
            raise ValueError(f"Error calculating LLM output match: {e}")
    
    
    def all_llm_output_exact_match(self):
        """
        This function calculates the percentage of LLM output 
        that exactly matches the expected output. It uses the LLM to 
        predict all medication names (informal names).
        
        Parameters:
        -----------
        None
        
        Returns:
        --------
        
        all_llm_output_exact_match_results: dict
            A dictionary containing the results of the all LLM output exact match.
            
        Raises:
        -------
        ValueError:
            If the columns 'Expected Output' and 'LLM All Predicted Name' are not in the DataFrame.
        """
        
        self.name = 'all_llm_output_exact_match'
        
        if 'Expected Output' not in self.df.columns or 'LLM All Predicted Name' not in self.df.columns:
            raise ValueError("Required columns ('Expected Output', 'LLM All Predicted Name') are not in the dataframe.")
        
        try:
        
            self.df['Expected Output'] = self.df['Expected Output'].str.strip().str.lower()
            self.df['LLM All Predicted Name'] = self.df['LLM All Predicted Name'].str.strip().str.lower()
            
            total_words = len(self.df)
            
            # Calculate the number of correct and wrong predictions of the LLM
            correct_predictions = (self.df['Expected Output'] == self.df['LLM All Predicted Name']).sum()
            wrong_predictions = total_words - correct_predictions
            
            # Calculate the percentage of correct and wrong predictions
            correct_percentage = (correct_predictions / total_words) * 100 if total_words > 0 else 0
            wrong_percentage = (wrong_predictions / total_words) * 100 if total_words > 0 else 0
            

            print(f'Total words: {total_words}')
            print(f'Correct LLM predictions: {correct_predictions}')
            print(f'Wrong LLM predictions: {wrong_predictions}')
            print(f'Correct LLM prediction percentage: {correct_percentage}%')
            print(f'Wrong LLM prediction percentage: {wrong_percentage}%')
            print("=" * 50)
            
            self.logger.update_log('all_llm_output_exact_match', {
                'total_words': total_words,
                'correct_predictions': correct_predictions,
                'wrong_predictions': wrong_predictions,
                'correct_percentage': correct_percentage,
                'wrong_percentage': wrong_percentage
            })
            
            return {
                'total_words': total_words,
                'correct_predictions': correct_predictions,
                'wrong_predictions': wrong_predictions,
                'correct_percentage': correct_percentage,
                'wrong_percentage': wrong_percentage
            }
        
        except Exception as e:
            raise ValueError(f"Error calculating all LLM output exact match: {e}")
        
        
    def all_llm_output_partial_match(self, threshold = 0.35):
        """
        This function calculates the percentage of LLM output that partially
        matches the expected output. It uses the LLM to predict all medication names
        (informal names) and also calculates the overall match percentage 
        including exact and partial matches.
        
        Parameters:
        -----------
        threshold: float
            The threshold value for partial match.
            
        Returns:
        --------
        all_llm_output_partial_match_results: dict
            A dictionary containing the results of the all LLM output partial match.
            
        Raises:
        -------
        ValueError:
            If the columns 'Expected Output' and 'LLM All Predicted Name' are not in the DataFrame.
        """
        
        self.name = 'all_llm_output_partial_match'
        
        if 'Expected Output' not in self.df.columns or 'LLM All Predicted Name' not in self.df.columns:
            raise ValueError("Required columns ('Expected Output', 'LLM All Predicted Name') are not in the dataframe.")
        
        try:
        
            self.df['Expected Output'] = self.df['Expected Output'].str.strip().str.lower()
            self.df['LLM All Predicted Name'] = self.df['LLM All Predicted Name'].str.strip().str.lower()

            total_words = len(self.df)
            correct_predictions = 0
            partial_matches = 0
            
            for index, row in self.df.iterrows():
                expected_output = row['Expected Output']
                predicted_name = row['LLM All Predicted Name']
                
                similarity = SequenceMatcher(None, expected_output, predicted_name).ratio()
                
                if similarity == 1.0:
                    correct_predictions += 1
                elif similarity >= threshold:
                    partial_matches += 1

            wrong_predictions = total_words - (correct_predictions + partial_matches)
            
            overall_including_partial = correct_predictions + partial_matches
            
            # Calculate the percentage of correct and partial matches
            correct_percentage = (correct_predictions / total_words) * 100 if total_words > 0 else 0
            partial_match_percentage = (partial_matches / total_words) * 100 if total_words > 0 else 0
            overall_including_partial_percentage = (overall_including_partial / total_words) * 100 if total_words > 0 else 0
            
        
            print(f'Total words: {total_words}')
            print(f'Correct LLM predictions: {correct_predictions}')
            print(f'Partial LLM predictions: {partial_matches}')
            print(f'Wrong LLM predictions: {wrong_predictions}')
            print(f'Overall LLM predictions (including partial): {overall_including_partial}')
            print(f'Correct LLM prediction percentage: {correct_percentage}%')
            print(f'Partial LLM prediction percentage: {partial_match_percentage}%')
            print(f'Overall (including partial) LLM prediction percentage: {overall_including_partial_percentage}%')
            print("=" * 50)
            
            self.logger.update_log('all_llm_output_partial_match', {
                'threshold': threshold,
                'total_words': total_words,
                'correct_predictions': correct_predictions,
                'partial_matches': partial_matches,
                'wrong_predictions': wrong_predictions,
                'correct_percentage': correct_percentage,
                'partial_match_percentage': partial_match_percentage,
                'overall_including_partial_percentage': overall_including_partial_percentage
            })
            
            return {
                'total_words': total_words,
                'correct_predictions': correct_predictions,
                'partial_matches': partial_matches,
                'wrong_predictions': wrong_predictions,
                'correct_percentage': correct_percentage,
                'partial_match_percentage': partial_match_percentage,
                'overall_including_partial_percentage': overall_including_partial_percentage
            }
        
        except Exception as e:
            raise ValueError(f"Error calculating all LLM output partial match: {e}")

    
    def calculate_accuracy_metrics(self):
        """
        This function calculates the overall accuracy of the predictions.
        
        Parameters:
        -----------
        None
        
        Returns:
        --------
        accuracy_metrics: dict
            A dictionary containing the results of the accuracy metrics.
            
        """
        
        self.name = 'calculate_accuracy_metrics'
        
        try:
            
            # Calculate the total correct matches
            informal_exact_match = self.df['Informal OMOP Match'].str.lower().eq('yes').sum()
            llm_exact_match = self.df.loc[self.df['Informal OMOP Match'].str.lower() == 'no', 'LLM OMOP Match'].str.lower().eq('yes').sum()
            
            
            total_correct_matches = informal_exact_match + llm_exact_match
            
            
            no_match = self.df.loc[
                (self.df['Informal OMOP Match'].str.lower() == 'no') & 
                (self.df['LLM OMOP Match'].str.lower() == 'no'), 
                'Informal OMOP Match'
            ].count()

        
            # Calculate the overall accuracy
            total_count = len(self.df)
            accuracy = (total_correct_matches / total_count) * 100 if total_count > 0 else 0
            no_match_percentage = (no_match / total_count) * 100 if total_count > 0 else 0
            

            print(f'Total Correct Matches: {total_correct_matches}')
            print(f'Overall Accuracy (%): {accuracy}')
            print(f'Total No Matches: {no_match}')
            print(f'No Match Percentage (%): {no_match_percentage}')
            print("=" * 50)
            
            self.logger.update_log('calculate_accuracy_metrics', {
                'total': total_count,
                'total_correct_matches': total_correct_matches,
                'total_no_matches': no_match,
                'accuracy_percentage': accuracy,
                'no_match_percentage': no_match_percentage,
            })
            
            return {
                'total': total_count,
                'total_correct_matches': total_correct_matches,
                'total_no_matches': no_match,
                'accuracy_percentage': accuracy,
                'no_match_percentage': no_match_percentage,
            }
        except Exception as e:
            raise ValueError(f"Error calculating accuracy metrics: {e}")

    
    def calculate_informal_omop_accuracy(self):
        """
        Calculate the accuracy based on the `Informal OMOP Match`.
        
        Parameters:
        -----------
        None
        
        Returns:
        --------
        informal_omop_accuracy: dict
            A dictionary containing the results of the informal OMOP accuracy.
            
        Raises:
        -------
        ValueError:
            If the column 'Informal OMOP Match' is not in the DataFrame.
        """
        self.name = 'calculate_informal_omop_accuracy'
        
        if 'Informal OMOP Match' not in self.df.columns:
            raise ValueError("Column 'Informal OMOP Match' is not in the dataframe.")
        
        try:
        
            correct_informal_match = self.df['Informal OMOP Match'].str.lower().eq('yes').sum()
            
            total_predictions = len(self.df)
            
            accuracy = (correct_informal_match / total_predictions) * 100 if total_predictions > 0 else 0
            
        
            print(f'Total Predictions: {total_predictions}')
            print(f'Informal OMOP Match Correct: {correct_informal_match}')
            print(f'Informal OMOP Match Accuracy: {accuracy:.2f}%')
            print("=" * 50)
            
            self.logger.update_log('calculate_informal_omop_accuracy', {
                'total': total_predictions,
                'correct_informal_match': correct_informal_match,
                'accuracy': accuracy
            })
        
            
            return {
                'total': total_predictions,
                'correct_informal_match': correct_informal_match,
                'accuracy': accuracy
            }
            
        except Exception as e:
            raise ValueError(f"Error calculating informal OMOP accuracy: {e}")

    
    
    def calculate_llm_omop_accuracy(self):
        """
        Calculate the accuracy based on the `LLM OMOP Match`.
        
        Parameters:
        -----------
        None
        
        Returns:
        --------
        llm_omop_accuracy: dict
            A dictionary containing the results of the LLM OMOP accuracy.
            
        Raises:
        -------
        ValueError:
            If the column 'LLM OMOP Match' is not in the DataFrame.
        """
        
        self.name = 'calculate_llm_omop_accuracy'
        
        if 'LLM OMOP Match' not in self.df.columns:
            raise ValueError("Column 'LLM OMOP Match' is not in the dataframe.")
        
        try:
        
            correct_llm_match = self.df['LLM OMOP Match'].str.lower().eq('yes').sum()
            
            total_predictions = len(self.df)
            
            accuracy = (correct_llm_match / total_predictions) * 100 if total_predictions > 0 else 0
            
        
            print(f'Total Predictions: {total_predictions}')
            print(f'LLM OMOP Match Correct: {correct_llm_match}')
            print(f'Informal OMOP Match Accuracy: {accuracy:.2f}%')
            print("=" * 50)
            
            self.logger.update_log('calculate_llm_omop_accuracy', {
                'total': total_predictions,
                'correct_llm_match': correct_llm_match,
                'accuracy': accuracy
            })
            
            return {
                'total': total_predictions,
                'correct_llm_match': correct_llm_match,
                'accuracy': accuracy
            }
        except Exception as e:
            raise ValueError(f"Error calculating LLM OMOP accuracy: {e}")


"""
# Test Case: 

if __name__ == "__main__":

    opt = EvaluationOptions().parse()

    concept_calculation = ConceptCalculation(opt)
    final_df = concept_calculation.run()

    metrics = Metrics(final_df)

    print("\n" + "=" * 50)
    print("Informal Name Match Results:")
    informal_match_results = metrics.informal_name_match()
    print(informal_match_results)
    print("=" * 50)
    
    print("\n" + "=" * 50)
    print("LLM Output Match Results:")
    llm_output_match_results = metrics.llm_output_match()
    print(llm_output_match_results)
    print("=" * 50)
    

    print("\n" + "=" * 50)
    print("All LLM Output Exact Match Results:")
    all_llm_output_exact_match_results = metrics.all_llm_output_exact_match()
    print(all_llm_output_exact_match_results)
    print("=" * 50)
    
    
    print("\n" + "=" * 50)
    print("All LLM Output Partial Match Results:")
    all_llm_output_partial_match_results = metrics.all_llm_output_partial_match()
    print(all_llm_output_partial_match_results)
    print("=" * 50)
    

    print("\n" + "=" * 50)
    print("Calculate Accuracy Metrics:")
    accuracy_metrics = metrics.calculate_accuracy_metrics()
    print(accuracy_metrics)
    print("=" * 50)
    
    print("\n" + "=" * 50)
    print("Calculate Informal OMOP Accuracy:")
    informal_omop_accuracy = metrics.calculate_informal_omop_accuracy()
    print(informal_omop_accuracy)
    print("=" * 50)
    
    print("\n" + "=" * 50)
    print("Calculate LLM OMOP Accuracy:")
    llm_omop_accuracy = metrics.calculate_llm_omop_accuracy()
    print(llm_omop_accuracy)
    print("=" * 50)
    
"""
    