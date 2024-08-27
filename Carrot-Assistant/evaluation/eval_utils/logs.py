import os
import json
from datetime import datetime

class EvaluationLogger:
    """
    This class is used to log information during the evaluation process.
    
    Parameters:
    -----------
    opt: EvaluationOptions
        The options for the evaluation.
    """

    def __init__(self, opt):
        self._opt = opt
        self.log_info = {
            "evaluation_start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics_evaluation": {},
        }

    def update_log(self, metric_name, result):
        """
        Update the log information with the given metric name and result.
        
        Parameters:
        -----------
        metric_name: str
            The name of the metric.
            
        result: float or dict
            The result of the evaluation metric.
        """
        self.log_info["metrics_evaluation"][metric_name] = result

    def save_pretty_log(self, directory=None):
        """
        Save the log information to a text file in a pretty format.
        The file name includes the date and time to ensure uniqueness.
        
        Parameters:
        -----------
        directory: str
            The directory to save the log file.
        """
        
        # If no directory is provided, use the default 'logs' directory
        if directory is None:
            directory = os.path.join("evaluation", "logs")
        
        # Ensure the directory exists
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Create a timestamp for the log file name
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"evaluation_log_{timestamp}.txt"
        file_path = os.path.join(directory, file_name)
        
        # Save the log file
        with open(file_path, "w") as f:
            f.write(self.format_pretty_log())
        print(f"Logs saved to {file_path}")

    def format_pretty_log(self):
        """
        Format the log information in a pretty format.
        
        Parameters:
        -----------
        None
        
        Returns:
        --------
        pretty_log: str
            The formatted log information
        """
        pretty_log = []
        pretty_log.append("=" * 90)
        pretty_log.append(
            " " * 10 + f"{self._opt.experiment_name} EVALUATION LOG" + " " * 10
        )
        pretty_log.append("=" * 90)
        pretty_log.append(f"\nEvaluation started at: {self.log_info['evaluation_start_time']}")
        pretty_log.append(
            f"\nModel Utilised for the Evaluation: {self._opt.llm_model}\n"
        )
        pretty_log.append("=" * 90)

        for metric_name, result in self.log_info["metrics_evaluation"].items():
            pretty_log.append(f"{metric_name.replace('_', ' ').upper()} RESULTS:\n")
            if isinstance(result, dict):
                for key, value in result.items():
                    pretty_log.append(f"  {key.replace('_', ' ').capitalize()}: {value}")
            pretty_log.append("-" * 80)

        pretty_log.append("=" * 90)

        return "\n".join(pretty_log)
