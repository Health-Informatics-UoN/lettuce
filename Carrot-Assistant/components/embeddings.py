from enum import Enum
from urllib.parse import quote_plus
from dotenv import load_dotenv
from haystack.dataclasses import Document
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.components.embedders.fastembed import FastembedDocumentEmbedder, FastembedTextEmbedder
import os
from os import environ
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from typing import Any, List, Dict
from pydantic import BaseModel

from omop.omop_models import Concept

class EmbeddingModelName(str, Enum):
    BGESMALL = "BGESMALL"
    MINILM = "MINILM"

class EmbeddingModelInfo(BaseModel):
    path: str
    dimensions: int

class EmbeddingModel(BaseModel):
    name: EmbeddingModelName
    info: EmbeddingModelInfo

EMBEDDING_MODELS = {
    EmbeddingModelName.BGESMALL: EmbeddingModelInfo(path="BAAI/bge-small-en-v1.5", dimensions=384),
    EmbeddingModelName.MINILM: EmbeddingModelInfo(path="sentence-transformers/all-MiniLM-L6-v2", dimensions=384),
}

def get_embedding_model(name: EmbeddingModelName) -> EmbeddingModel:
    return EmbeddingModel(name=name, info=EMBEDDING_MODELS[name])

class Embeddings:
    """
    This class allows the building or loading of a vector database of concept names. This database can then be used for vector search.

    Methods
    -------
    search:
        Query the attached embeddings database with provided search terms
    """
    def __init__(
            self,
            embeddings_path: str,
            force_rebuild: bool,
            embed_vocab: List[str],
            model_name: EmbeddingModelName,
            search_kwargs: dict,
            ) -> None:
        """
        Initialises the connection to an embeddings database

        Parameters
        ----------
        embeddings_path: str
            A path for the embeddings database. If one is not found, it will be built, which takes a long time. This is built from concepts fetched from the OMOP database
        force_rebuild: bool
            If true, the embeddings database will be rebuilt.
        embed_vocab: List[str]
            A list of OMOP vocabulary_ids. If the embeddings database is built, these will be the vocabularies used in the OMOP query
        model: EmbeddingModel
            The model used to create embeddings
        search_kwargs: dict
            kwargs for vector search
        """
        self.embeddings_path = embeddings_path
        self.model = get_embedding_model(model_name)
        self.embed_vocab = embed_vocab
        self.search_kwargs = search_kwargs
        
        if force_rebuild or not os.path.exists(embeddings_path):
            self._build_embeddings()
        else:
            self._load_embeddings()

    def _build_embeddings(self):
        """
        Build a vector database of embeddings
        """
        # Create the directory if it doesn't exist
        if os.path.dirname(self.embeddings_path):
            os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)

        self.embeddings_store = QdrantDocumentStore(
            path=self.embeddings_path,
            embedding_dim=self.model.info.dimensions,
            recreate_index=True  # We're building from scratch, so recreate the index
        )
        
        load_dotenv()

        DB_HOST = os.environ["DB_HOST"]
        DB_USER = environ["DB_USER"]
        DB_PASSWORD = quote_plus(environ["DB_PASSWORD"])
        DB_NAME = environ["DB_NAME"]
        DB_PORT = environ["DB_PORT"]

        connection_string = (
            f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )

        # Fetch concept names from the database
        engine = create_engine(connection_string)
        with Session(engine) as session:
            concepts = session.query(Concept.concept_name, Concept.concept_id).filter(
                Concept.vocabulary_id.in_(self.embed_vocab)
            ).all()
        
        # Create documents from concept names
        concept_docs = [Document(
            content = concept.concept_name,
            meta = {"concept_id": concept.concept_id}
            ) for concept in concepts]
        concept_embedder = FastembedDocumentEmbedder(model=self.model.info.path, parallel=0)
        concept_embedder.warm_up()
        concept_embeddings = concept_embedder.run(concept_docs)
        self.embeddings_store.write_documents(concept_embeddings.get("documents"))

    def _load_embeddings(self):
        """
        If available, load a vector database of concept embeddings
        """
        self.embeddings_store = QdrantDocumentStore(
            path=self.embeddings_path,
            embedding_dim=self.model.info.dimensions,
            recreate_index=False  # We're loading existing embeddings, don't recreate
        )

    def search(self, query: List[str]) -> List[List[Dict[str,Any]]]:
        """
        Search the attached vector database with a list of informal medications

        Parameters
        ----------
        query: List[str]
            A list of informal medication names

        Returns
        -------
        List[List[Dict[str, Any]]]
            For each medication in the query, the result of searching the vector database
        """
        retriever = QdrantEmbeddingRetriever(
            document_store=self.embeddings_store
        )
        query_embedder = FastembedTextEmbedder(model=self.model.info.path, parallel=0, prefix="query:")
        query_embedder.warm_up()
        query_embeddings = [query_embedder.run(name) for name in query]
        result = [retriever.run(query_embedding['embedding'], **self.search_kwargs) for query_embedding in query_embeddings]
        return [[{"concept_id": doc.meta["concept_id"], "concept": doc.content, "score": doc.score} for doc in res["documents"]] for res in result]