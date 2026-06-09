# DB utils

## Classes
### `PGConnector`
```python
class PGConnector:
    db_user: str,
    db_password: str,
    db_host: str,
    db_port: int,
    db_name: str,
    db_schema: str,
    logger: Logger,
    embeddings_table_name: str = "embeddings",
```

Configures a connection to a postgresql database with pgvector

| Parameter | Type | Description |
| --------------- | --------------- | --------------- |
| `db_user` | str | The user for the postgres connection URL |
| `db_password` | str | The password for the postgres connection URL |
| `db_host` | str | The host for the postgres connection URL |
| `db_port` | int | The port for the postgres connection URL |
| `db_name` | str | The database name for the postgres connection URL |
| `db_schema` | str | The name of the database schema to use |
| `logger` | logging.Logger | Logging for database interactions |
| `embeddings_table_name` | str | The name for the embeddings table in the specified schema |

| Property | Description |
| -------- | ----------- |
| `db_schema` | The name of the database schema used |
| `embeddings_table_name` | The name of the embeddings table |

#### Methods
##### `check_extension`
```python
check_extension() -> None:
```

Check whether the connected database has the pgvector extension installed

##### `reset_embedding_table`
```python
reset_embedding_table() -> None:
```

Drop any existing embedding table, then create a new one with the right vector dimension

| Parameter | Type | Description |
| --------------- | --------------- | --------------- |
| `embedding_dimension` | int | The table needs a specified dimension for the embeddings |

##### `get_connection`
```python
get_connection() -> psycopg.Connection
```

Return a configured connection

