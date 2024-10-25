import pandas as pd

from evaluation.evaltypes import EvalDataLoader


class SingleInputSimpleCSV(EvalDataLoader):
    """
    Implements the EvalDataLoader class to load single results from a csv file.
    The file must have a column named 'input_data', each entry defining an input to a pipeline, and another column 'expected_output', defining the desired output for the matching input.
    """

    def __init__(self, file_path: str) -> None:
        """
        Initialises a SingleInputSimpleCSV

        Parameters
        ----------
        file_path: str
            A path pointing to the input file
        """
        super().__init__(file_path)
        self.data = pd.read_csv(file_path)
        self._input_data = self.load_input_data()
        self._expected_output = self.load_expected_output()

    @property
    def input_data(self) -> list:
        """
        The input_data for an EvaluationFramework
        """
        return self._input_data

    @property
    def expected_output(self) -> list:
        """
        The expected_output for an EvaluationFramework
        """
        return self._expected_output

    def load_input_data(self) -> list:
        """
        Loads the input data column from the specified file

        Returns
        -------
        list
            A list of the input_data column
        """
        try:
            return [[i] for i in self.data["input_data"]]
        except KeyError:
            print(f"No column named 'input_data' in {self.file_path}")

    def load_expected_output(self) -> list:
        """
        Loads the expected output column from the specified column

        Returns
        -------
        list
            A list of the expected_output column
        """
        try:
            return list(self.data["expected_output"])
        except KeyError:
            print(f"No column named 'expected_output' in {self.file_path}")


class SingleInputCSVforLLM(EvalDataLoader):
    """
    Implements the EvalDataLoader class to load single results from a csv file.
    The file must have a column named 'input_data', each entry defining an input to a pipeline, and another column 'expected_output', defining the desired output for the matching input.
    The data loader splits the input_data into a list of lists for compatibility with LLM pipelines
    """

    def __init__(self, file_path: str) -> None:
        """
        Initialises a SingleInputCSVforLLM

        Parameters
        ----------
        file_path: str
            A path pointing to the input file
        """
        super().__init__(file_path)
        self.data = pd.read_csv(file_path)
        self._input_data = self.load_input_data()
        self._expected_output = self.load_expected_output()

    @property
    def input_data(self) -> list:
        """
        The input_data for an EvaluationFramework
        """
        return self._input_data

    @property
    def expected_output(self) -> list:
        """
        The expected_output for an EvaluationFramework
        """
        return self._expected_output

    def load_input_data(self) -> list:
        """
        Loads the input data column from the specified file

        Returns
        -------
        list
            A list of the input_data column, where each item is a length 1 list for compatibility with LLMPipelines
        """
        try:
            return [[i] for i in self.data["input_data"]]
        except KeyError:
            print(f"No column named 'input_data' in {self.file_path}")

    def load_expected_output(self) -> list:
        """
        Loads the expected output column from the specified column

        Returns
        -------
        list
            A list of the expected_output column
        """
        try:
            return list(self.data["expected_output"])
        except KeyError:
            print(f"No column named 'expected_output' in {self.file_path}")
