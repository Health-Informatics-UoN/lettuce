from evaluation.eval_data_loaders import SingleInputCSVforLLM
from evaluation.evaltypes import EvaluationFramework
from evaluation.metrics import UncasedMatch, FuzzyMatchRatio
from evaluation.pipelines import LLMPipeline
from options.pipeline_options import LLMModel
from evaluation.eval_tests import LLMPipelineTest
from jinja2 import Environment


dataloader = SingleInputCSVforLLM("./evaluation/datasets/example.csv")

jinja_env = Environment()
prompt_template = jinja_env.from_string(
    """You will be given the informal name of a medication. Respond only with the formal name of that medication, without any extra explanation.

Examples:

Informal name: Tylenol
Response: Acetaminophen

Informal name: Advil
Response: Ibuprofen

Informal name: Motrin
Response: Ibuprofen

Informal name: Aleve
Response: Naproxen

Task:

Informal name: {{informal_name}}<|eot_id|>
Response:"""
)

template_vars = ["informal_name"]

pipelines = [
    (
        "Llama 2 7b",
        LLMPipeline(
            llm=LLMModel.LLAMA_2_7B,
            prompt_template=prompt_template,
            template_vars=template_vars,
        ),
    ),
    (
        "Kuchiki",
        LLMPipeline(
            llm=LLMModel.KUCHIKI_L2_7B,
            prompt_template=prompt_template,
            template_vars=template_vars,
        ),
    ),
    (
        "BioMistral 7b",
        LLMPipeline(
            llm=LLMModel.BIOMISTRAL_7B,
            prompt_template=prompt_template,
            template_vars=template_vars,
        ),
    ),
    (
        "Airobouros 3B",
        LLMPipeline(
            llm=LLMModel.AIROBOROS_3B,
            prompt_template=prompt_template,
            template_vars=template_vars,
        ),
    ),
    (
        "Llama 3.1 8b",
        LLMPipeline(
            llm=LLMModel.LLAMA_3_1_8B,
            prompt_template=prompt_template,
            template_vars=template_vars,
        ),
    ),
]

pipeline_tests = [
    LLMPipelineTest(name, pipeline, [UncasedMatch(), FuzzyMatchRatio()])
    for name, pipeline in pipelines
]

if __name__ == "__main__":
    evaluation = EvaluationFramework(
        name="Example benchmarking",
        pipeline_tests=pipeline_tests,
        dataset=dataloader,
        description="Demonstration of running LLMs against a benchmark, applying two metrics. The example data is the first 40 entries from the HELIOS self-reported medications dataset",
        results_path="example_output.json",
    )

    evaluation.run_evaluations()
