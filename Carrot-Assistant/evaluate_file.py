from evaluation.eval_data_loaders import SingleInputCSVforLLM
from evaluation.evaltypes import EvaluationFramework
from evaluation.metrics import ExactMatchMetric
from evaluation.pipelines import LLMPipeline
from options.pipeline_options import LLMModel
from evaluation.eval_tests import LLMPipelineTest
from jinja2 import Environment


dataloader = SingleInputCSVforLLM("./evaluation/datasets/example.csv")

exact_match = ExactMatchMetric()

jinja_env = Environment()
l_3_1_prompt_template = jinja_env.from_string(
    """
You will be given the informal name of a medication. Respond only with the formal name of that medication, without any extra explanation.

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
Response:
                                              """
)

other_template = jinja_env.from_string(
    """
You will be given the informal name of a medication. Respond only with the formal name of that medication, without any extra explanation.

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

Informal name: {{informal_name}}
Response:
                                       """
)

llama_3_1_pipeline = LLMPipeline(
    llm=LLMModel.LLAMA_3_1_8B,
    prompt_template=l_3_1_prompt_template,
    template_vars=["informal_name"],
)

llama_3_pipeline = LLMPipeline(
    llm=LLMModel.LLAMA_3_8B,
    prompt_template=other_template,
    template_vars=["informal_name"],
)

llama_3_1_test = LLMPipelineTest(
    name="Llama 3.1 8B", pipeline=llama_3_1_pipeline, metrics=[ExactMatchMetric()]
)
llama_3_test = LLMPipelineTest(
    name="Llama 3 8B", pipeline=llama_3_pipeline, metrics=[ExactMatchMetric()]
)

if __name__ == "__main__":
    evaluation = EvaluationFramework(
        name="Example experiment",
        pipeline_tests=[llama_3_1_test],
        dataset=dataloader,
        description="A small example experiment, comparing Llama 3 and Llama 3.1 8B models on the same prompt",
        results_path="example_output.json",
    )

    evaluation.run_evaluations()
