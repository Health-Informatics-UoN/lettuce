import os
import sys
from options.base_options import BaseOptions

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class EvaluationOptions(BaseOptions):

    def __init__(self) -> None:
        super().__init__()

    def initialize(self) -> None:
        """Initializes the TrainOptions class"""

        BaseOptions.initialize(self)
        
        self._parser.add_argument(
            "--experiment_name",
            type=str,
            required=False,
            default="Llettuce Medication Name Evaluation",
            help="Name of the experiment",
        )

        # Change the root according to the location of the file
        self._parser.add_argument(
            "--process_file_path",
            type=str,
            default="/Carrot-Assistant/evaluation/PartialTestingOMOPConcepts.xlsx",
            required=False,
            help="Path to the file to be processed",
        )

        self._parser.add_argument(
            "--informal_name_column",
            type=str,
            default="Medication",
            required=False,
            help="Name of the column containing the informal medication names",
        )

        self._parser.add_argument(
            "--expected_results_column",
            type=str,
            default="OMOP concept name",
            required=False,
            help="Name of the column containing the expected OMOP concept names",
        )

        self._parser.add_argument(
            "--metrics",
            type=str,
            nargs="+",
            default=[
                "informal_name_match",
                "llm_output_match",
                "all_llm_output_exact_match",
                "all_llm_output_partial_match",
                "calculate_accuracy_metrics",
                "calculate_informal_omop_accuracy",
                "calculate_llm_omop_accuracy",
            ],
            choices=[
                "informal_name_match",
                "llm_output_match",
                "all_llm_output_exact_match",
                "all_llm_output_partial_match",
                "calculate_accuracy_metrics",
                "calculate_informal_omop_accuracy",
                "calculate_llm_omop_accuracy",
            ],
            help="List of metrics to be calculated",
        )

        # Change the root according to the location of the file
        self._parser.add_argument(
            "--log_path",
            type=str,
            default="/Carrot-Assistant/evaluation/logs",
            required=False,
            help="Path to the log file",
        )