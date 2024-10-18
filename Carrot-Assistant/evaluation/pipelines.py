from typing import Dict
from evaltypes import SingleResultPipeline
from options.pipeline_options import LLMModel
from components.models import local_models
from jinja2 import Template
from llama_cpp import Llama
from huggingface_hub import hf_hub_download


class LLMPipeline(SingleResultPipeline):
    def __init__(self, llm: LLMModel, prompt_template: Template) -> None:
        self.llm = llm
        self.prompt_template = prompt_template
        self._model = Llama(hf_hub_download(**local_models[self.llm.value]))

    def run(self, input: Dict[str, str]) -> str:
        prompt = self.prompt_template.render(input)
        return self._model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}]
        )["choices"][0]["message"]
