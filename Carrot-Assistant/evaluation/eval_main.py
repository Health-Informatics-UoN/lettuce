import os
import sys
from datetime import datetime

from options.eval_options import EvaluationOptions
from evaluation.concept_calculation import ConceptCalculation
from evaluation.call_methods import make_evaluation
from evaluation.eval_utils.logs import EvaluationLogger  

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run():
    """
    Run the evaluation process.
    
    Parameters:
    -----------
    
    Returns:
    --------
    None
    
    Process:
    --------
    1. Parse the evaluation options.
    2. Initialize the logger.
    3. Run the concept calculation.
    4. Get the metrics to run.
    5. Run each metric and log the results.
    6. Log the evaluation end.
    7. Save the log in a pretty format.
    """
    # Parse the evaluation options
    opt = EvaluationOptions().parse()

    # Initialize the logger
    logger = EvaluationLogger(opt)
    logger.update_log('evaluation_start', {'status': 'Evaluation started'})

    # Run the concept calculation
    concept_calculation = ConceptCalculation(opt)
    final_df = concept_calculation.run()

    # Get the metrics to run
    metrics_to_run = opt.metrics
    logger.update_log('Evaluation Metrics Used', {'Metrics': metrics_to_run})

    if not metrics_to_run:
        raise ValueError("No metrics provided")

    # Run each metric and log the results
    for metric_name in metrics_to_run:
        logger.update_log('running_metric', {'metric': metric_name})
        
        print("\n" + "=" * 50)
        print(f"\nRunning metric: {metric_name}")
        
        # Pass the logger to the metric evaluation
        result = make_evaluation(metric_name, final_df, logger=logger)
        print(result)
        
        # Log the result of each metric
        logger.update_log(metric_name, result)
    
    logger.update_log('evaluation_end', {'status': 'Evaluation completed'})
    
    print("\nEvaluation completed.\n")

    logger.save_pretty_log()


if __name__ == "__main__":
    run()
