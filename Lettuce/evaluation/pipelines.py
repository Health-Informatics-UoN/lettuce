from sentence_transformers import SentenceTransformer
from torch.functional import Tensor
from evaluation.evaltypes import SingleResultPipeline
from jinja2 import Template
from llama_cpp import Llama
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever


class LLMPipeline(SingleResultPipeline):
    """
    This class runs a simple LLM-only pipeline on provided input
    """

    def __init__(
        self, llm: Llama, prompt_template: Template, template_vars: list[str]
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
        self.prompt_template = prompt_template
        self._model = llm
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
        print(f"Replied {reply} for {input}")
        return reply

    def drop(self):
        del self._model


class EmbeddingsPipeline(SingleResultPipeline):

    def __init__(self, embedding_model: SentenceTransformer) -> None:
        self.model = embedding_model

    def run(self, input: str) -> Tensor:
        return self.model.encode(input)


class RAGPipeline(SingleResultPipeline):
    def __init__(
        self,
        llm: Llama,
        prompt_template: Template,
        template_vars: list[str],
        embedding_model: SentenceTransformer,
        retriever: QdrantEmbeddingRetriever,
        top_k: int = 5,
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template
        self._llmodel = llm
        self._embedding_model = embedding_model
        self._template_vars = template_vars
        self._retriever = retriever
        self._top_k = top_k

    def run(self, input: list[str]) -> str:
        embedding = self._embedding_model.encode(input[0])
        search_results = self._retriever.run(embedding, top_k=self._top_k)
        prompt = self.prompt_template.render(
            dict(zip(self._template_vars, [*input, search_results["documents"]]))
        )
        reply = self._llmodel.create_completion(prompt=prompt)["choices"][0]["text"]
        print(f"Replied {reply} for {input}")
        return reply
