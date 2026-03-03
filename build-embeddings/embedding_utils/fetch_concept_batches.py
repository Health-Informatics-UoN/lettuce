from collections.abc import Generator

from embedding_utils.protocols import (
    ConceptEmbedder,
    ConceptReader,
    EmbeddedConcept,
    EmbeddingPipeline,
    EmbeddingStore,
)


class BatchEmbeddingPipeline(EmbeddingPipeline):
    """
    An Embedding Pipeline that reads, embeds, and loads batches of concepts

    Attributes
    ----------
    reader: ConceptReader
        A ConceptReader that extracts concept records and can generate batches of Concepts
    embedder: ConceptEmbedder
        A ConceptEmbedder that can take batches of Concepts and encode them, returning batches of EmbeddedConcepts
    store: EmbeddingStore
        An EmbeddingStore that can take batches of Embedded concepts and store them somewhere
    """
    def __init__(
        self, reader: ConceptReader, embedder: ConceptEmbedder, store: EmbeddingStore
    ) -> None:
        self.reader = reader
        self.embedder = embedder
        self.store = store

    def embed_batches(self) -> Generator[list[EmbeddedConcept]]:
        """
        Yields EmbeddedConcepts using the Concept Generator from the reader and passing them through the embedder

        Yields
        ------
        list[EmbeddedConcept]
        """
        return (
            self.embedder.embed_concepts(batch)
            for batch in self.reader.load_concept_batch()
        )

    def run_pipeline(self) -> None:
        """
        Run the complete pipeline, loading batches of concepts
        """
        for batch in self.embed_batches():
            self.store.save(batch)


class AllConceptEmbeddingPipeline(EmbeddingPipeline):
    """
    An Embedding Pipeline that reads, embeds, and loads a complete set of concepts

    Attributes
    ----------
    reader: ConceptReader
        A ConceptReader that extracts concept records and returns them as concepts
    embedder: ConceptEmbedder
        A ConceptEmbedder that can take the list of Concepts and encode them, returning them as EmbeddedConcepts
    store: EmbeddingStore
        An EmbeddingStore that can take the set of embedded concepts and store them somewhere
    """
    def __init__(
        self, reader: ConceptReader, embedder: ConceptEmbedder, store: EmbeddingStore
    ) -> None:
        self.reader = reader
        self.embedder = embedder
        self.store = store

    def run_pipeline(self) -> None:
        """
        Run the complete pipeline, extracting and loading the full set of concepts
        """
        concepts = self.reader.load_concepts()
        embeddings = self.embedder.embed_concepts(concepts)
        self.store.save(embeddings)
