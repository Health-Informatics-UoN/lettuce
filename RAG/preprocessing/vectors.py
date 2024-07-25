from typing import Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from preprocessing.embeddings_const import get_embeddings

import os
from constant import PACKAGE_ROOT_PATH
from utils.utils import load_folder_contents


def csv_folder_retriever(opt, main_data_folder, use_multithreading):
    local_vector_store = LocalVectorStore(opt)
    if opt.use_local_vector_store and local_vector_store.is_available():
        print("Using local vector store")
        main_retriever = local_vector_store.get_store()

    else:
        print("Creating vector store")
        # Load CSV documents
        main_documents = load_folder_contents(
            path=main_data_folder,
            glob="**/*.csv",
            doc_type="csv",
            loader_kwargs={"csv_args": {"delimiter": "\t"}},
            use_multithreading=use_multithreading,
        )

        # process documents
        vs_processor = VectorStoreProcessor(opt)
        main_retriever = vs_processor.get_store(main_documents)
    return main_retriever


class LocalVectorStore:
    def __init__(self, opt):
        self._opt = opt
        self._embedding_model = opt.embedding_model["model"]
        self._embedding_model_name = opt.embedding_model["model_name"]
        self._vector_store_name = opt.vector_store
        self._no_doc_to_retrieve = opt.no_doc_to_retrieve
        self._embedding = get_embeddings(
            self._embedding_model, self._embedding_model_name
        )

    def is_available(self):
        if self._opt.use_local_vector_store:
            self._store_path = (
                PACKAGE_ROOT_PATH
                / ".cache"
                / self._vector_store_name
                / self._embedding_model
                / self._embedding_model_name
            )

            if self._vector_store_name.lower() == "faiss":
                if os.path.exists(self._store_path / "index.faiss") and os.path.exists(
                    self._store_path / "index.pkl"
                ):
                    return True
                else:
                    return False

            elif self._vector_store_name.lower() == "chroma":
                if os.path.exists(self._store_path / "chroma.sqlite3"):
                    return True
                else:
                    return False
        else:
            return False

    def get_store(self):
        if self._vector_store_name.lower() == "faiss":
            self._vector_store = FAISS.load_local(
                str(self._store_path), embeddings=self._embedding
            )

        elif self._vector_store_name.lower() == "chroma":
            self._vector_store = Chroma(
                persist_directory=str(self._store_path),
                embedding_function=self._embedding,
            )
        self._retriever = self._vector_store.as_retriever(
            search_kwargs={"k": self._no_doc_to_retrieve}
        )
        return self._retriever


class VectorStoreProcessor:
    def __init__(self, opt):
        self._opt = opt
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=opt.chunk_size, chunk_overlap=opt.chunk_overlap
        )
        self._embedding_model = opt.embedding_model["model"]
        self._embedding_model_name = opt.embedding_model["model_name"]
        self._vector_store_name = opt.vector_store
        self._no_doc_to_retrieve = opt.no_doc_to_retrieve

    def _split_documents(self, documents):
        return self._splitter.split_documents(documents)

    def _make_vector_store(self, documents):
        store = LocalFileStore(
            PACKAGE_ROOT_PATH
            / ".cache"
            / self._vector_store_name
            / self._embedding_model
            / self._embedding_model_name
        )
        underlying_embeddings = self._embedding
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings, store, namespace="local_"
        )
        if self._vector_store_name.lower() == "faiss":
            self._vector_store = FAISS.from_documents(documents, cached_embedder)
            self._vector_store.save_local(str(store.root_path))

        elif self._vector_store_name.lower() == "chroma":
            self._vector_store = Chroma.from_documents(
                documents, cached_embedder, persist_directory=str(store.root_path)
            )
            self._vector_store.persist()

        else:
            raise NotImplementedError(
                f"Vector store {self._vector_store_name} not supported"
            )

        return self._vector_store

    def get_store(self, documents):
        documents = self._split_documents(documents)
        self._embedding = get_embeddings(
            self._embedding_model, self._embedding_model_name
        )
        self._vector_store = self._make_vector_store(documents)
        self._retriever = self._vector_store.as_retriever(
            search_kwargs={"k": self._no_doc_to_retrieve}
        )
        return self._retriever
