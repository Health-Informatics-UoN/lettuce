from collections.abc import Callable, Generator
from logging import Logger
from pathlib import Path
from jinja2 import Template
from torch import Tensor
import psycopg as pg
from psycopg import sql
import polars as pl
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from embedding_utils.protocols import ConceptEmbedder, ConceptReader, EmbeddedConcept, EmbeddingPipeline, EmbeddingStore
from embedding_utils.string_building import Concept
from embedding_utils.save_embedding import copy_to_postgres

class BatchEmbeddingPipeline(EmbeddingPipeline):
    def __init__(
            self,
            reader: ConceptReader,
            embedder: ConceptEmbedder,
            store: EmbeddingStore
            ) -> None:
        self.reader = reader
        self.embedder = embedder
        self.store = store

    def embed_batches(self) -> Generator[list[EmbeddedConcept]]:
        return (self.embedder.embed_concepts(batch) for batch in self.reader.load_concept_batch())

    def run_pipeline(self) -> None:
        for batch in self.embed_batches():
            self.store.save(batch)

class AllConceptEmbeddingPipeline(EmbeddingPipeline):
    def __init__(
            self,
            reader: ConceptReader,
            embedder: ConceptEmbedder,
            store: EmbeddingStore
            ) -> None:
        self.reader = reader
        self.embedder = embedder
        self.store = store

    def run_pipeline(self) -> None:
        concepts = self.reader.load_concepts()
        embeddings = self.embedder.embed_concepts(concepts)
        self.store.save(embeddings)

class EmbeddingPipelineFactory:
    def __init__(self) -> None:
        pass

    def add_reader(self, reader: ConceptReader) -> None:
        self.get_concept_reader = reader

    def add_embedder(self, embedder: ConceptEmbedder) -> None:
        self.get_concept_embedder = embedder

    def add_store(self, store: EmbeddingStore) -> None:
        self.get_embedding_store = store

    def create_pipeline(
            self,
            batch_pipeline: bool
            ) -> EmbeddingPipeline:
        if batch_pipeline:
            return BatchEmbeddingPipeline(
                    reader=self.get_concept_reader,
                    embedder=self.get_concept_embedder,
                    store=self.get_embedding_store
                    )
        else:
            return AllConceptEmbeddingPipeline(
                    reader=self.get_concept_reader,
                    embedder=self.get_concept_embedder,
                    store=self.get_embedding_store
                    )
