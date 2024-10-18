from typing import Dict
from evaluation.evaltypes import SingleResultPipeline
from options.pipeline_options import LLMModel
from components.models import local_models
from jinja2 import Template
from llama_cpp import Llama
from huggingface_hub import hf_hub_download


class LLMPipeline(SingleResultPipeline):
    """
    This class runs a simple LLM-only pipeline on provided input
    """

    def __init__(self, llm: LLMModel, prompt_template: Template) -> None:
        """
        Initialises the LLMPipeline class

        Parameters
        ----------
        llm: LLMModel
            One of the model options in the LLMModel enum
        prompt_template: Template
            A jinja2 template for a prompt
        """
        self.llm = llm
        self.prompt_template = prompt_template
        self._model = Llama(hf_hub_download(**local_models[self.llm.value]))

    def run(self, input: Dict[str, str]) -> str:
        """
        Runs the LLMPipeline on a given input

        Parameters
        ----------
        input: Dict[str, str]
            The input is rendered into a prompt string by the .render method of the prompt template, so needs to be a dictionary of the template's parameters

        Returns
        -------
        str
            The output of running the prompt through the given model
        """
        prompt = self.prompt_template.render(input)
        return self._model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}]
        )["choices"][0]["message"]["content"]
