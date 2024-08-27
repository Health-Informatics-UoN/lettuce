import pandas as pd


class ProcessTestDataset:
    """
    This class is used to process the test dataset for the evaluation of the model.
    
    Parameters:
    -----------
    
    results_file: str
        The name of the file to store the results of the evaluation.
    """

    def __init__(self, results_file="results.json"):
        self.results_file = results_file

    def load_test_dataset(self, file_path: str, header: int = 0):
        """
        This function is used to load the test dataset for the evaluation of the model.
        
        Parameters:
        -----------
        file_path: str
            The path to the test dataset.
            
        header: int
            The row number to use as the column names.
            
        Returns:
        --------
        df: pd.DataFrame
            The DataFrame containing the test dataset.
            
        Raises:
        -------
            ValueError: If the file path is not provided.
        """

        if not file_path:
            raise ValueError("File path not provided")

        try:
            df = pd.read_excel(file_path, header=header)
            print(df.head())
            return df

        except Exception as e:
            raise ValueError(f"Error loading dataset: {e}")

    def extract_medication_names(self, df: pd.DataFrame, informal_name_column: str):
        """
        This function is used to extract the medication names from the test dataset.
        
        Parameters:
        -----------
        df: pd.DataFrame
            The DataFrame containing the test dataset.
        
        informal_name_column: str
            The column name containing the medication names.
            
        Returns:
        --------
        informal_names: list
            The list of medication names extracted from the test dataset.
            
        Raises:
        -------
            ValueError: If the medication column is not found in the DataFrame.
        """

        if not informal_name_column in df.columns:
            raise ValueError("Medication column not found in DataFrame")

        if informal_name_column in df.columns:

            try:
                informal_names = df[informal_name_column].dropna().tolist()
                print(f"\nExtracted medication names: {informal_names[:5]}\n")
                return informal_names

            except Exception as e:
                raise ValueError(f"Error extracting medication names: {e}")

    def extract_expected_results(self, df: pd.DataFrame, expected_results_column: str):
        """
        This function is used to extract the expected results from the test dataset.
        
        Parameters:
        -----------
        df: pd.DataFrame
            The DataFrame containing the test dataset.
            
        expected_results_column: str
            The column name containing the expected results.
            
        Returns:
        --------
        expected_results: list
            The list of expected results extracted from the test dataset.
            
        Raises:
        -------
            ValueError: If the OMOP Concept Name column is not found in the DataFrame.
        """

        if not expected_results_column in df.columns:
            raise ValueError("OMOP Concept Name column not found in DataFrame")

        if expected_results_column in df.columns:

            try:
                expected_results = df[expected_results_column].dropna().tolist()
                print(f"\nExtracted expected results: {expected_results[:5]}\n")
                return expected_results

            except Exception as e:
                raise ValueError(f"Error extracting expected results: {e}")

    def run_process(
        self, file_path: str, informal_name_column: str, expected_results_column: str
    ):
        """
        This function is used to run the process of extracting the 
        medication names and expected results from the test dataset.
        
        Parameters:
        -----------
        file_path: str
            The path to the test dataset.
            
        informal_name_column: str
            The column name containing the medication names.
            
        expected_results_column: str
            The column name containing the expected results.
            
        Returns:
        --------
        informal_names: list
            The list of medication names extracted from the test dataset.
            
        expected_results: list
            The list of expected results extracted from the test dataset.
        """
        df = self.load_test_dataset(file_path)
        informal_names = self.extract_medication_names(
            df, informal_name_column=informal_name_column
        )
        expected_results = self.extract_expected_results(
            df, expected_results_column=expected_results_column
        )
        
        processed_df = df.copy()
        processed_df['Informal Name'] = processed_df.pop('Medication')
        processed_df['Expected Output'] = processed_df.pop('OMOP concept name')
        
        
        print(f"\nProcessed DataFrame: {processed_df.head()}\n")
        
        return informal_names, expected_results, processed_df


"""
# Test usage
if __name__ == "__main__":
    process = ProcessTestDataset()
    medication_names, expected_results, processed_df = process.run_process(
        file_path='/Users/karthik/lettuce-UI-Update-QueryFirst-UI-LLM-Option/Carrot-Assistant/evaluation/PartialTestingOMOPConcepts.xlsx',
        informal_name_column="Medication",
        expected_results_column="OMOP concept name",
    )
"""

