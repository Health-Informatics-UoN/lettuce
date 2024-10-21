import pandas as pd

from evaluation.evaltypes import EvalDataLoader


class SingleInputSimpleCSV(EvalDataLoader):
    def __init__(self, file_path) -> None:
        super().__init__(file_path)
        self.data = pd.read_csv(file_path)
        try:
            self.input_data = self.data["input_data"]
        except KeyError:
            print(f"No column named 'input_data' in {self.file_path}")
        try:
            self.expected_output = self.data["expected_output"]
        except KeyError:
            print(f"No column named 'expected_output' in {self.file_path}")
