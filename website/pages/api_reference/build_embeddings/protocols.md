# Protocols

Embeddings pipelines require the extraction of concepts from some source, encoding these as embeddings, and loading this to some output.

To manage this, there are protocols that should make integration of new sources, embedders, and sinks.

## Classes
### `EmbeddingPipeline`
```python
class EmbeddingPipeline(Protocol):
    reader: ConceptReader
    embedder: ConceptEmbedder
    store: EmbeddingStore
```

Protocol for a pipeline that reads, encodes, and loads concepts

- `ConceptReader`s have to expose two methods: `load_concepts`, which returns a list of [`Concept`s](/api_reference/build_embeddings/string_building), and `load_concept_batch`, which returns a Generator for a list of `Concept`s.
- `ConceptEmbedder`s have to expose a method, `embed_concepts`, which takes a list of `Concept`s and returns a list of `EmbeddedConcept`s.
- `EmbeddingStore`s have to expose a method, `save`, which takes a list of `EmbeddedConcept`s and doesn't return anything.

This means that an `EmbeddingPipeline` can use `reader` to fetch (a batch of) concepts, feed these through the `embedder` and use the `store` to save them.
Batched and complete pipelines are implemented in [fetch_concept_batches](./fetch_concept_batches).

#### Methods
##### `run_pipeline`
```python
run_pipeline() -> None
```

Run the embedding pipeline.

### `ConceptReader`
```python
class ConceptReader(Protocol):
    _batch_size: int
``` 
Protocol for a concept reader that can either take a full set of concepts or read in batches

#### Methods
##### `load_concept_batch`
```python
load_concept_batch() -> Generator[list[Concept]]
```

Return a Generator to iterate through loaded concepts in batches.

##### `load_concepts`
```python
load_concepts() -> list[Concept]
```

### `ConceptEmbedder`
```python
class ConceptEmbedder(Protocol)
```

Protocol for a thing that can take concepts and produce embeddings

#### Methods
##### `embed_concepts`
```python
embed_concepts(concepts: list[Concept]) -> list[EmbeddedConcept]
```
Take a list of concepts and encode them into embeddings

### `EmbeddingStore`
```python
class EmbeddingStore(Protocol)
```

Protocol for taking embeddings and storing them somewhere

#### Methods
##### `save`
```python
save(embeddings: list[EmbeddedConcept]) -> None
```

Take a list of embeddings and save them somewhere.

### `EmbeddedConcept`
```python
class EmbeddedConcept:
concept_id: int
concept_name: str 
embedding: list[float]
```

Dataclass to hold identifiers for a concept and its embedding
