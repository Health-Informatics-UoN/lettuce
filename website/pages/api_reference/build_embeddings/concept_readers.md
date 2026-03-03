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

Returns a Generator for batches of Concepts from the database

###### Yields `Generator[list[Concept]]`

`batch_size` concepts from the database

##### `load_concepts`
```python
load_concepts() -> list[Concept]
```

Returns the concepts from the database.
This could be several million.

###### Returns `list[Concept]`
All the concepts from the database

### `CsvConceptExtractor`
```python
class CsvConceptExtractor(ConceptReader):
    path: Path
    batch_size: int
    table_schema: dict[str, polars.DataType] = CONCEPT_SCHEMA
```

A ConceptReader that connects to a CSV file in the Athena standard format and reads the concepts.

| Attribute | Type | Description |
| --------- | ---- | ----------- |
| path | Path | The path to a CSV file of concepts |
| batch_size | int | The number of concepts to retrieve at a time | 
| table_schema | dict[str, polars.DataType] | A schema for the concept table. The default works for files from Athena |

#### Methods
```python
load_concept_batch() -> Generator[list[Concept]]
```

Returns a Generator for batches of Concepts from the file

###### Yields `Generator[list[Concept]]`

`batch_size` concepts from the database

##### `load_concepts`
```python
load_concepts() -> list[Concept]
```

Returns the concepts from the file.
This could be several million.

###### Returns `list[Concept]`
All the concepts from the file

## Functions
### `parse_rows`
```python
def parse_rows(rows: list[tuple[int, str, str, str, str]]) -> list[Concept]:
``` 

Take rows from the concept table and build a list of `Concepts` for them

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| rows  | `list[tuple[int, str, str, str, str]]` | A row from the concepts table |

#### Returns `list[Concept]` 
The concepts as `Concept` objects


## Constants
### `CONCEPT_SCHEMA`
A schema for polars to parse the CSV files downloaded from Athena
