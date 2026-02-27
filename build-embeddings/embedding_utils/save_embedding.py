import polars as pl
from pathlib import Path
from psycopg import sql
import time

from embedding_utils.db_utils import PGConnector
from embedding_utils.protocols import EmbeddedConcept, EmbeddingStore


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
            db_connector: PGConnector,
            ) -> None:
        super().__init__()
        self._db_connector = db_connector

    def save(self, embeddings: list[EmbeddedConcept]):
        with self._db_connector.get_connection().cursor("embed cursor") as embed_cursor:
            with embed_cursor.copy(
                    sql.SQL(
                        "COPY {} (concept_id, embedding) FROM STDIN WITH (FORMAT BINARY)"
                        ).format(
                            sql.Identifier(
                                self._db_connector.db_schema,
                                self._db_connector.embeddings_table_name
                                )
                            )
                    ) as copy:
                copy.set_types(["int4", "vector"])
                for entry in embeddings:
                    copy.write_row((entry.concept_id, entry.embedding))
