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
from typing import List

from omop.omop_models import Concept

class EmbeddingModel(Enum):
    BGESMALL = "BAAI/bge-small-en-v1.5", 384
    MINILM = "sentence-transformers/all-MiniLM-L6-v2", 384
    
    def __init__(self, path: str, dimensions: int) -> None:
        self.path = path
        self.dimensions = dimensions

class Embeddings:
    def __init__(
            self,
            embeddings_path: str,
            force_rebuild: bool,
            embed_vocab: List[str],
            model: EmbeddingModel,
            search_kwargs: dict,
            ) -> None:
        self.embeddings_path = embeddings_path
        self.model = model
        self.embed_vocab = embed_vocab
        self.search_kwargs = search_kwargs
        
        if force_rebuild or not os.path.exists(embeddings_path):
            self._build_embeddings()
        else:
            self._load_embeddings()

    def _build_embeddings(self):
        # Create the directory if it doesn't exist
        if os.path.dirname(self.embeddings_path):
            os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)

        self.embeddings_store = QdrantDocumentStore(
            path=self.embeddings_path,
            embedding_dim=self.model.dimensions,
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
        concept_embedder = FastembedDocumentEmbedder(model=self.model.path, parallel=0)
        concept_embedder.warm_up()
        concept_embeddings = concept_embedder.run(concept_docs)
        self.embeddings_store.write_documents(concept_embeddings.get("documents"))

    def _load_embeddings(self):
        self.embeddings_store = QdrantDocumentStore(
            path=self.embeddings_path,
            embedding_dim=self.model.dimensions,
            recreate_index=False  # We're loading existing embeddings, don't recreate
        )

    def search(self, query: str, top_k: int = 5):
        retriever = QdrantEmbeddingRetriever(
            document_store=self.embeddings_store
        )
        query_embedder = FastembedTextEmbedder(model=self.model.path, parallel=0, prefix="query:")
        query_embedding = query_embedder.run(query)
        return retriever.run(query_embedding, **self.search_kwargs)
