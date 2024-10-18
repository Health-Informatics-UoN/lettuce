from evaltypes import SingleResultPipeline
from options.pipeline_options import LLMModel
from jinja2 import Template


class LLMPipeline(SingleResultPipeline):
    def __init__(self, llm: LLMModel, prompt_template: Template) -> None:
        self.llm = (LLMModel,)
        self.prompt_template = prompt_template

    def run(self, input) -> str:
        pass
