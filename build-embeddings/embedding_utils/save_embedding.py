import polars as pl
from pathlib import Path
from psycopg import sql
import time

from embedding_utils.db_utils import PGConnector
from embedding_utils.protocols import EmbeddedConcept, EmbeddingStore


class ParquetWriter(EmbeddingStore):
    """
    An EmbeddingStore that saves batches of concepts to a parquet file

    Attributes
    ----------
    path: Path
        The path to which embedding batches are saved
    """

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path

    def save(self, embeddings: list[EmbeddedConcept]) -> None:
        """
        Saves batches of embeddings to the writer's path.
        The extra timestamp column allows the writer to append batches to the file because the `partition_by` behaviour handles multiple datasets.
        You can still treat it as a single dataframe and drop the timestamp when using it.

        Parameters
        ----------
        embeddings: list[EmbeddedConcept]
            A list of concepts with embeddings
        """
        pl.DataFrame(
            {
                "timestamp": time.time(),
                "concept_id": [c.concept_id for c in embeddings],
                "concept_name": [c.concept_name for c in embeddings],
                "embeddings": [c.embedding for c in embeddings],
            }
        ).write_parquet(self._path, partition_by="timestamp")


class PostgresWriter(EmbeddingStore):
    """
    An EmbeddingStore that loads batches of concepts in a postgres database

    Attributes
    ----------
    db_connector: PGConnector
        A configured connection
    """

    def __init__(
        self,
        db_connector: PGConnector,
    ) -> None:
        super().__init__()
        self._db_connector = db_connector

    def save(self, embeddings: list[EmbeddedConcept]) -> None:
        """
        Saves batches of embeddings to the configured database
        
        Parameters
        ---------
        embeddings: list[EmbeddedConcept]
            A list of concepts with embeddings
        """
        with self._db_connector.get_connection() as conn:
            with conn.cursor("embed cursor") as embed_cursor:
                with embed_cursor.copy(
                    sql.SQL(
                        "COPY {} (concept_id, embedding) FROM STDIN WITH (FORMAT BINARY)"
                    ).format(
                        sql.Identifier(
                            self._db_connector.db_schema,
                            self._db_connector.embeddings_table_name,
                        )
                    )
                ) as copy:
                    copy.set_types(["int4", "vector"])
                    for entry in embeddings:
                        copy.write_row((entry.concept_id, entry.embedding))
                    self._db_connector._logger.info(f"Written batch of {len(embeddings)} concepts")
            conn.commit()
            
