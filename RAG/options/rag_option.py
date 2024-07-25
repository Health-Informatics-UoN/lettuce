import ast
import os
import sys
from typing import Dict

from options.base_option import BaseOptions


class RAGOptions(BaseOptions):
    """RAG Options"""

    def __init__(self) -> None:
        super().__init__()

    def initialize(self) -> None:
        BaseOptions.initialize(self)

        self._parser.add_argument(
            "--problem_type",
            type=str,
            required=False,
            default="csv_agent",
            choices=["csv_folder_retriever", "csv_agent"],
            help="problem type",
        )

        self._parser.add_argument(
            "--embedding_model",
            type=lambda x: ast.literal_eval(x),
            required=False,
            default={"model": "OpenAI", "model_name": "text-embedding-3-small"},
            choices=[
                {"OpenAI", "text-embedding-3-small"},
                {"HuggingFace", "hkunlp/instructor-xl"},
            ],
            help="embedding model",
        )

        self._parser.add_argument(
            "--vector_store",
            type=str,
            required=False,
            default="FAISS",
            choices=["FAISS", "Chroma"],
            help="vector store",
        )

        self._parser.add_argument(
            "--use_local_vector_store",  # TODO: This flag can be removed in future so the app check for the local vector store by default
            type=bool,
            required=False,
            default=True,
            help="use local vector store",
        )

        self._parser.add_argument(
            "--no_doc_to_retrieve",
            type=int,
            required=False,
            default=10,
            help="number of documents to retrieve",
        )

        self._parser.add_argument(
            "--task",
            type=lambda x: ast.literal_eval(x),
            required=False,
            default={"task_type": "retrieval_chat"},
            choices=[{"retrieval_chat"}],
            help="task type",
        )

        self._parser.add_argument(
            "--llm_model",
            type=lambda x: ast.literal_eval(x),
            required=False,
            default={"model": "OpenAI", "model_name": "gpt-4"},
            choices=[
                {"OpenAI", "gpt-3.5-turbo-0125"},
                {"OpenAI", "gpt-4"},
            ],
        )

        self._parser.add_argument(
            "--llm_temperature",
            type=float,
            required=False,
            default=0.0,
            help="llm temperature to control the model's creativity, higher values give more creative results. range: 0.0-1.0",
        )
