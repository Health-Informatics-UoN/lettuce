import sys
import os
import pandas as pd
from evaluation.process import ProcessTestDataset
from evaluation.concept_extraction import OmopConceptNameTesting

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ConceptCalculation:
    """
    This class is used to calculate the concepts from the test dataset.
    
    Parameters:
    -----------
    opt: EvaluationOptions
        The options for the evaluation.
    """
    def __init__(self, opt):
        self._opt = opt

        self.informal_names = None
        self.expected_results = None
        self.processed_df = None
        self.results = None
        self.llm_results = None
        self.formal_name_results = None

    def process_dataset(self):
        """
        This function is used to process the test dataset for the evaluation of the model.
        
        Parameters:
        -----------
        None
        
        Raises:
        -------
        ValueError: 
            If there is an error processing the dataset
        
        """
        if (
            self.informal_names is None
            or self.expected_results is None
            or self.processed_df is None
        ):
            try:
                process = ProcessTestDataset()
                self.informal_names, self.expected_results, self.processed_df = (
                    process.run_process(
                        file_path=self._opt.process_file_path,
                        informal_name_column=self._opt.informal_name_column,
                        expected_results_column=self._opt.expected_results_column,
                    )
                )
                print("\nStep 1 Completed: Dataset Processed\n")
            except Exception as e:
                raise ValueError(f"Error processing dataset: {e}")

    def omop_concept_testing(self):
        """
        This function is used to run the OMOP concept name testing on the test dataset.
        
        Parameters:
        -----------
        None
        
        Raises:
        -------
        ValueError: 
            If there is an error running the OMOP concept name testing.
        """
        if (
            self.results is None
            or self.llm_results is None
            or self.formal_name_results is None
        ):
            try:
                tester = OmopConceptNameTesting()
                (
                    self.results,
                    self.llm_results,
                    self.formal_name_results,
                    self.llm_all_results,
                ) = tester.run(self.informal_names)

                print("\nStep 2 Completed: OMOP Concept Testing\n")
            except Exception as e:
                raise ValueError(f"Error running OMOP concept name testing: {e}")

    def concat_dataframe(self):
        """
        This function is used to concatenate the processed DataFrame with 
        the OMOP concept testing results.
        
        Parameters:
        -----------
        None
        
        Raises:
        -------
        ValueError: 
            If there is an error concatenating the DataFrames.
        """
        if (
            self.results is None
            or self.llm_results is None
            or self.formal_name_results is None
        ):
            raise ValueError(
                "You must run OMOP concept testing before concatenating dataframes."
            )

        tester = OmopConceptNameTesting()
        concept_extracted_df = tester.create_concept_extracted_dataframe(
            self.results, self.llm_results, self.formal_name_results, self.llm_all_results 
        )

        print("Columns in processed_df:", self.processed_df.columns)
        print("Columns in concept_extracted_df:", concept_extracted_df.columns)

        final_df = pd.merge(
            self.processed_df, concept_extracted_df, on="Informal Name", how="left"
        )

        print("\nStep 3 Completed: DataFrames Combined\n")
        print("\n" + "=" * 50)
        print("Final Combined DataFrame:")
        print("=" * 50)
        print(final_df)
        print("=" * 50)

        return final_df

    def run(self):
        """
        This function is used to run the ConceptCalculation.
        
        Parameters:
        -----------
        None
        
        Returns:
        --------
        final_df: pd.DataFrame
            The final DataFrame containing the test dataset with the OMOP concept testing results.
        """
        self.process_dataset()
        self.omop_concept_testing()
        final_df = self.concat_dataframe()
        return final_df

"""
# Testcases:
if __name__ == "__main__":

    opt = EvaluationOptions().parse()

    # Run the ConceptCalculation
    concept_calculation = ConceptCalculation(opt)
    final_df = concept_calculation.run()
"""