from typing import Protocol
from collections.abc import Generator
from dataclasses import dataclass

from embedding_utils.string_building import Concept


@dataclass
class EmbeddedConcept:
    """
    Dataclass to hold identifiers for a concept and its embedding
    """
    concept_id: int
    concept_name: str
    embedding: list[float]


class EmbeddingStore(Protocol):
    """Protocol for taking embeddings and storing them somewhere"""
    def save(self, embeddings: list[EmbeddedConcept]) -> None: ...


class ConceptReader(Protocol):
    """Protocol for a concept reader that can either take a full set of concepts or read in batches"""
    _batch_size: int

    def load_concept_batch(self) -> Generator[list[Concept]]: ...

    def load_concepts(self) -> list[Concept]: ...


class ConceptEmbedder(Protocol):
    """Protocol for a thing that can take concepts and produce embeddings"""
    def embed_concepts(self, concepts: list[Concept]) -> list[EmbeddedConcept]: ...


class EmbeddingPipeline(Protocol):
    """Protocol for a pipeline that reads, encodes, and loads concepts"""
    reader: ConceptReader
    embedder: ConceptEmbedder
    store: EmbeddingStore

    def run_pipeline(self) -> None: ...
