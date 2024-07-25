from llm.models import get_model
from llm.prompts import Prompts
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import ConversationalRetrievalChain, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser


class Chains:
    def __init__(self, opt, retriever):
        self._opt = opt
        self._retriever = retriever
        self._task_type = opt.task["task_type"]
        self._llm_model = opt.llm_model["model"]
        self._llm_model_name = opt.llm_model["model_name"]
        self._llm_temperature = opt.llm_temperature

        self._llm = get_model(
            self._llm_model, self._llm_model_name, temperature=self._llm_temperature
        )

    def get_chain(self):
        if self._task_type.lower() == "retrieval_chat":
            self._prompt = Prompts().get_prompt("simple_retrieval")
            return self._retrieval_chat()

    def _retrieval_chat(self):
        chain = (
            {
                "context": itemgetter("question") | self._retriever,
                "question": itemgetter("question"),
            }
            | self._prompt
            | self._llm
            | StrOutputParser()
        )
        return chain
