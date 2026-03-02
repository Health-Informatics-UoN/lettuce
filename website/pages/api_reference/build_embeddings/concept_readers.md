# Concept Readers

Running an [embeddings building pipeline](./protocols) requires a component that can read [`Concept`s](./string_building) from input.

## Classes
### `PostgresConceptExtractor`
```python
class PostgresConceptExtractor(ConceptReader):
    db_connector: PGConnector
    batch_size: int
    logger: Logger
```

A [`ConceptReader`](./protocols) that connects to a PostgreSQL database and reads the concepts.

| Attribute | Description |
|-----------|-------------|
| `db_connector` | A [`PGConnector`](./db_utils) that can configure a connection to a database |
| `batch_size` | An integer specifying the size of batches to fetch from the database |
| `logger` | log database interactions |

#### Methods
##### `load_concept_batch`
```python
load_concept_batch() -> Generator[list[Concept]]
```

##### `load_concepts`
```python
load_concepts() -> list[Concept]
```

### `CsvConceptExtractor`

#### Methods
##### `load_concept_batch`
```python
load_concept_batch() -> Generator[list[Concept]]
```

##### `load_concepts`
```python
load_concepts() -> list[Concept]
```

## Functions
### `parse_rows`

## Constants
### `CONCEPT_SCHEMA`
