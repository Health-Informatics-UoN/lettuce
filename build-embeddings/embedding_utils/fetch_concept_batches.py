from collections.abc import Generator

from embedding_utils.protocols import (
    ConceptEmbedder,
    ConceptReader,
    EmbeddedConcept,
    EmbeddingPipeline,
    EmbeddingStore,
)


class BatchEmbeddingPipeline(EmbeddingPipeline):
    def __init__(
        self, reader: ConceptReader, embedder: ConceptEmbedder, store: EmbeddingStore
    ) -> None:
        self.reader = reader
        self.embedder = embedder
        self.store = store

    def embed_batches(self) -> Generator[list[EmbeddedConcept]]:
        return (
            self.embedder.embed_concepts(batch)
            for batch in self.reader.load_concept_batch()
        )

    def run_pipeline(self) -> None:
        for batch in self.embed_batches():
            self.store.save(batch)


class AllConceptEmbeddingPipeline(EmbeddingPipeline):
    def __init__(
        self, reader: ConceptReader, embedder: ConceptEmbedder, store: EmbeddingStore
    ) -> None:
        self.reader = reader
        self.embedder = embedder
        self.store = store

    def run_pipeline(self) -> None:
        concepts = self.reader.load_concepts()
        embeddings = self.embedder.embed_concepts(concepts)
        self.store.save(embeddings)
