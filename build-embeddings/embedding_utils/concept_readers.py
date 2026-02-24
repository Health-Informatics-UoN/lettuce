from collections.abc import Generator
from logging import Logger

from psycopg import sql

from embedding_utils.string_building import Concept
from embedding_utils.protocols import ConceptReader
from embedding_utils.db_utils import PGConnector

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
            logger: Logger
            ) -> None:
        self.db_connector = db_connector
        self.logger = logger
        self._concept_query = sql.SQL("SELECT concept_id, concept_name, domain_id, vocabulary_id, concept_class_id FROM {}").format(sql.Identifier(self.db_connector.db_schema, "concept"))
        
    def load_concept_batch(self, batch_size: int) -> Generator[list[Concept]]:
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
                    rows = concept_cursor.fetchmany(batch_size)
                    if len(rows) > 0:
                        yield [
                                Concept(
                                    concept_id=r[0],
                                    concept_name=r[1],
                                    domain=r[2],
                                    vocabulary=r[3],
                                    concept_class=r[4]
                                    ) for r in rows
                                ]

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
                        return [
                                Concept(
                                    concept_id=r[0],
                                    concept_name=r[1],
                                    domain=r[2],
                                    vocabulary=r[3],
                                    concept_class=r[4]
                                    ) for r in rows
                                ]
