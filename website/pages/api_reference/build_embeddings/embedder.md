# Embedder

## Classes

### `BatchEmbedder`
```python
class BatchEmbedder(ConceptEmbedder):
   embedding_model: SentenceTransformer,
   template: Template,
```

A ConceptEmbedder that can take a template and an embeddings model to produce concepts and their embeddings.

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| `embedding_model` | SentenceTransformer | The SentenceTransformer used to produce embeddings |
| `template` | Template | A jinja2 template to render strings for the embedding model to encode |

| Property | Description |
| -------- | ----------- |
| dimension | The embedding dimension of the attached model |

#### Methods
##### `embed_concepts`
```python
def embed_concepts(self, concepts: list[Concept]) -> list[EmbeddedConcept]:
```
Render the strings, encode the strings

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| `concepts` | `list[Concept]` | A list of concepts to encode |

###### Returns `list[EmbeddedConcepts]`
A list of encoded concepts
