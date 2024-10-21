import pandas as pd

from evaluation.evaltypes import EvalDataLoader


class SingleInputSimpleCSV(EvalDataLoader):
    def __init__(self, file_path) -> None:
        super().__init__(file_path)
        self.data = pd.read_csv(file_path)
        self._input_data = self.load_input_data()
        self._expected_output = self.load_expected_output()

    @property
    def input_data(self) -> list:
        return self._input_data

    @property
    def expected_output(self) -> list:
        return self._expected_output

    def load_input_data(self) -> list:
        try:
            return list(self.data["input_data"])
        except KeyError:
            print(f"No column named 'input_data' in {self.file_path}")

    def load_expected_output(self) -> list:
        try:
            return list(self.data["expected_output"])
        except KeyError:
            print(f"No column named 'expected_output' in {self.file_path}")
