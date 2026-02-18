import polars as pl
from pathlib import Path
from torch import Tensor
import psycopg as pg
from psycopg import sql
import time

from embedding_utils.protocols import EmbeddedConcept, EmbeddingStore

def save_parquet(path: Path, concepts: list[tuple[int, str, Tensor]]):
    pl.DataFrame(
            {
                "concept_id": [c[0] for c in concepts],
                "concept_name": [c[1] for c in concepts],
                "embeddings": [c[2] for c in concepts],
                }
            ).write_parquet(path)

def copy_to_postgres(cursor: pg.Cursor, concepts: list[EmbeddedConcept], db_schema: str, embedding_table_name: str) -> None:
    with cursor.copy(
            sql.SQL("COPY {} (concept_id, embedding) FROM STDIN WITH (FORMAT BINARY)").format(sql.Identifier(db_schema, embedding_table_name))
            ) as copy:
        copy.set_types(["int4", "vector"])
        for entry in concepts:
            copy.write_row((entry.concept_id, entry.embedding))

class ParquetWriter(EmbeddingStore):
    def __init__(
            self,
            path: Path
            ) -> None:
        super().__init__()
        self._path = path

    def save(self, embeddings: list[EmbeddedConcept]):
        pl.DataFrame({
            "timestamp": time.time(),
            "concept_id": [c.concept_id for c in embeddings],
            "concept_name": [c.concept_name for c in embeddings],
            "embeddings": [c.embedding for c in embeddings]
            }).write_parquet(self._path, partition_by="timestamp")

class PostgresWriter(EmbeddingStore):
    def __init__(
            self,
            connection: pg.Connection,
            db_schema: str,
            embedding_table_name: str,
            ) -> None:
        super().__init__()
        self._connection = connection
        self._db_schema = db_schema
        self._embedding_table_name = embedding_table_name

    def save(self, embeddings: list[EmbeddedConcept]):
        with self._connection.cursor("embed cursor") as embed_cursor:
            copy_to_postgres(embed_cursor, embeddings, self._db_schema, self._embedding_table_name)
