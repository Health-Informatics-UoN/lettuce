# Save embeddings

## Classes
### `ParquetWriter`
```python
class ParquetWriter(EmbeddingStore):
    path: Path
```

An EmbeddingStore that saves batches of concepts to a parquet file

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| `path` | Path | The path to which embedding batches are saved |

#### Methods
##### `save`
```python
save(embeddings: list[EmbeddedConcept]) -> None:
```

Saves batches of embeddings to the writer's path.
The extra timestamp column allows the writer to append batches to the file because the `partition_by` behaviour handles multiple datasets.
You can still treat it as a single dataframe and drop the timestamp when using it.

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| `embeddings` | list[EmbeddedConcept] | A list of concepts with embeddings |


### `PostgresWriter`
```python
class PostgresWriter(EmbeddingStore):
    db_connector: PGConnector,
```
An EmbeddingStore that loads batches of concepts in a postgres database

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| `db_connector` | PGConnector | A configured connection |

#### Methods
##### `save`
```python
save(embeddings: list[EmbeddedConcept]) -> None:
```
Saves batches of embeddings to the configured database

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| `embeddings` | list[EmbeddedConcept] | A list of concepts with embeddings |
