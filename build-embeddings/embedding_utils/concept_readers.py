from collections.abc import Generator
from logging import Logger
from pathlib import Path

from psycopg import sql
import polars as pl

from embedding_utils.string_building import Concept
from embedding_utils.protocols import ConceptReader
from embedding_utils.db_utils import PGConnector

# Polars' schema inference doesn't work well on Athena's vocab files, so it needs an explicit schema
CONCEPT_SCHEMA = {
        "concept_id": pl.Int32(),
        "concept_name": pl.String(),
        "domain_id": pl.String(),
        "vocabulary_id": pl.String(),
        "concept_class_id": pl.String(),
        "standard_concept": pl.String(),
        "concept_code": pl.String(),
        "valid_start_date": pl.String(),
        "valid_end_date": pl.String(),
        "invalid_reason": pl.String(),
        }

def parse_rows(rows: list[tuple[int, str, str, str, str]]) -> list[Concept]:
    """
    Take rows from the concept table and build a list of Concepts for them

    Parameters
    ----------
    rows: list
        A list of rows from the database

    Returns
    -------
    list[Concept]
    """
    return [
              Concept(
                  concept_id=r[0],
                  concept_name=r[1],
                  domain=r[2],
                  vocabulary=r[3],
                  concept_class=r[4]
                  ) for r in rows
            ]

class PostgresConceptExtractor(ConceptReader):
    """
    A ConceptReader that connects to a postgres database and reads the concepts.

    Attributes
    ----------
    db_connector: PGConnector
        A configured connection to a postgresql database 
    logger: Logger
    """
    def __init__(
            self,
            db_connector: PGConnector,
            batch_size: int,
            logger: Logger
            ) -> None:
        self.db_connector = db_connector
        self.logger = logger
        self._concept_query = sql.SQL("SELECT concept_id, concept_name, domain_id, vocabulary_id, concept_class_id FROM {}").format(sql.Identifier(self.db_connector.db_schema, "concept"))
        self._batch_size = batch_size
        
    def load_concept_batch(self) -> Generator[list[Concept]]:
        """
        Returns a Generator for batches of Concepts from the database

        Parameters
        ----------
        batch_size: int
            The number of concepts to fetch from the database

        Yields
        ------
        Generator[list[Concept]]
            batch_size concepts from the database
        """
        with self.db_connector.get_connection() as conn:
            with conn.cursor("concept cursor") as concept_cursor:
                while True:
                    concept_cursor.execute(self._concept_query)
                    rows = concept_cursor.fetchmany(self._batch_size)
                    if len(rows) > 0:
                        yield parse_rows(rows)

    def load_concepts(self) -> list[Concept]:
        """
        Returns the concepts from the database.
        This could be several million.

        Returns
        -------
        list[Concept]
        """
        with self.db_connector.get_connection() as conn:
            with conn.cursor("concept cursor") as concept_cursor:
                while True:
                    concept_cursor.execute(self._concept_query)
                    rows = concept_cursor.fetchall()
                    if len(rows) > 0:
                        return parse_rows(rows)

class CsvConceptExtractor(ConceptReader):
    """
    A ConceptReader that connects to a CSV file in the Athena standard format and reads the concepts.

    Attributes
    ----------
    path: Path
        The path to a CSV file of concepts
    table_schema: dict[str, pl.DataType]
        A schema for the concept table
    """
    def __init__(
            self,
            path: Path,
            batch_size: int,
            table_schema: dict[str, pl.DataType] = CONCEPT_SCHEMA,
            ) -> None:
        self._path = path
        self._table_schema = table_schema
        self._batch_size = batch_size

    def load_concept_batch(self) -> Generator[list[Concept]]:
        concept_df = pl.scan_csv(self._path, schema=self._table_schema, separator='\t', quote_char=None)
        for batch in concept_df.collect_batches(chunk_size=self._batch_size):
            yield [
                    Concept(
                        concept_id=r["concept_id"],
                        concept_name=r["concept_name"],
                        domain=r["domain_id"],
                        vocabulary=r["vocabulary_id"],
                        concept_class=r["concept_class_id"]
                        ) for r in batch.iter_rows(named=True)
                    ]

    def load_concepts(self) -> list[Concept]:
        concept_df = pl.read_csv(self._path, schema=self._table_schema, separator='\t', quote_char=None)
        return [
                Concept(
                    concept_id=r["concept_id"],
                    concept_name=r["concept_name"],
                    domain=r["domain_id"],
                    vocabulary=r["vocabulary_id"],
                    concept_class=r["concept_class_id"]
                    ) for r in concept_df.iter_rows(named=True)
                ]
