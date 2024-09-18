import sys
import os


def make_evaluation(metric_name: str, *args, **kwargs):
    """
    This function is used to import the Metrics class 
    and call the required method for evaluation.
    
    Parameters:
    -----------
    metric_name: str
        The name of the metric to be used for evaluation.
        
    *args:
        The arguments to be passed to the Metrics class.
        
    **kwargs:
        The keyword arguments to be passed to the Metrics class.
        
    Returns:
    --------
    result: float
        The result of the evaluation metric.
        
    Raises:
    -------
    ValueError: 
        If the metric name is invalid.
    """


    from evaluation.metrics import Metrics
    metrics = Metrics(*args, **kwargs)  

    if metric_name.lower() == "informal_name_match":
        return metrics.informal_name_match()

    elif metric_name.lower() == "llm_output_match":
        return metrics.llm_output_match()

    elif metric_name.lower() == "all_llm_output_exact_match":
        return metrics.all_llm_output_exact_match()

    elif metric_name.lower() == "all_llm_output_partial_match":
        return metrics.all_llm_output_partial_match()

    elif metric_name.lower() == "calculate_accuracy_metrics":
        return metrics.calculate_accuracy_metrics()

    elif metric_name.lower() == "calculate_informal_omop_accuracy":
        return metrics.calculate_informal_omop_accuracy()

    elif metric_name.lower() == "calculate_llm_omop_accuracy":
        return metrics.calculate_llm_omop_accuracy()

    else:
        raise ValueError("Invalid metric name")
        print(f"\nMetric: {metric_name} used for evaluation\n")
