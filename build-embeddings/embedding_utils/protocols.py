from typing import Protocol
from collections.abc import Generator
from dataclasses import dataclass

from embedding_utils.string_building import Concept

@dataclass
class EmbeddedConcept:
    concept_id: int
    concept_name: str
    embedding: list[float]

class EmbeddingStore(Protocol):
    def save(self, embeddings: list[EmbeddedConcept]) -> None:
        ...

class ConceptReader(Protocol):
    def load_concept_batch(self, batch_size: int) -> Generator[list[Concept]]:
        ...

    def load_concepts(self) -> list[Concept]:
        ...

class ConceptEmbedder(Protocol):
    def embed_concepts(self, concepts: list[Concept]) -> list[EmbeddedConcept]:
        ...

class EmbeddingPipeline(Protocol):
    reader: ConceptReader
    embedder: ConceptEmbedder
    store: EmbeddingStore
    
    def run_pipeline(self) -> None:
        ...
