from sentence_transformers import SentenceTransformer
from torch.functional import Tensor
from evaluation.evaltypes import SingleResultPipeline
from options.pipeline_options import LLMModel
from components.models import local_models
from jinja2 import Template
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import numpy as np


class LLMPipeline(SingleResultPipeline):
    """
    This class runs a simple LLM-only pipeline on provided input
    """

    def __init__(
        self, llm: LLMModel, prompt_template: Template, template_vars: list[str]
    ) -> None:
        """
        Initialises the LLMPipeline class

        Parameters
        ----------
        llm: LLMModel
            One of the model options in the LLMModel enum
        prompt_template: Template
            A jinja2 template for a prompt
        template_vars: list[str]
            The variables inserted into the prompt template when rendered
        """
        self.llm = llm
        self.prompt_template = prompt_template
        self._model = Llama(
            hf_hub_download(**local_models[self.llm.value]),
            n_ctx=0,
            n_batch=512,
            model_kwargs={"n_gpu_layers": -1, "verbose": True},
            generation_kwargs={"max_tokens": 128, "temperature": 0},
        )
        self._template_vars = template_vars

    def run(self, input: list[str]) -> str:
        """
        Runs the LLMPipeline on a given input

        Parameters
        ----------
        input: list[str]
            The input strings passed to the prompt template, in the order the template_vars were provided to the class

        Returns
        -------
        str
            The output of running the prompt through the given model
        """
        prompt = self.prompt_template.render(
            {(v, i) for v, i in zip(self._template_vars, input)}
        )
        reply = self._model.create_completion(prompt=prompt)["choices"][0]["text"]
        print(f"{self.llm.value} replied {reply} for {input}")
        return reply

    def drop(self):
        del self._model


class EmbeddingsPipeline(SingleResultPipeline):

    def __init__(self, embedding_model: SentenceTransformer) -> None:
        self.model = embedding_model

    def run(self, input: str) -> Tensor:
        return self.model.encode(input)
