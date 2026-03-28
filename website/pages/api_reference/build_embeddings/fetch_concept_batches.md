# Fetch Concept Batches

## Classes
### `BatchEmbeddingPipeline`

```python
class BatchEmbeddingPipeline(EmbeddingPipeline):
   self.reader = reader
   self.embedder = embedder
   self.store = store
```

An Embedding Pipeline that reads, embeds, and loads batches of concepts

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| `reader` | ConceptReader | A ConceptReader that extracts concept records and can generate batches of Concepts |
| `embedder` | ConceptEmbedder | A ConceptEmbedder that can take batches of Concepts and encode them, returning batches of EmbeddedConcepts |
| `store` | EmbeddingStore | An EmbeddingStore that can take batches of Embedded concepts and store them somewhere |

#### Methods
##### `embed_batches`
```python
embed_batches() -> Generator[list[EmbeddedConcept]]:
```

Yields EmbeddedConcepts using the Concept Generator from the reader and passing them through the embedder

###### Yields `list[EmbeddedConcept]`
Generator for batches of [`EmbeddedConcept`s](./protocols)

##### `run_pipeline`
```python
run_pipeline() -> None:
```

Run the complete pipeline, loading batches of concepts to the store

### `AllConceptEmbeddingPipeline`
```python
class AllConceptEmbeddingPipeline(EmbeddingPipeline):
    reader: ConceptReader
    embedder: ConceptEmbedder
    store: EmbeddingStore
```

An Embedding Pipeline that reads, embeds, and loads a complete set of concepts

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| `reader` | ConceptReader | A ConceptReader that extracts concept records and returns them as concepts |
| `embedder` | ConceptEmbedder | A ConceptEmbedder that can take the list of Concepts and encode them, returning them as EmbeddedConcepts |
| `store` | EmbeddingStore | An EmbeddingStore that can take the set of embedded concepts and store them somewhere |

#### Methods
##### `run_pipeline`
```python
run_pipeline() -> None:
```
Run the complete pipeline, extracting and loading the full set of concepts
