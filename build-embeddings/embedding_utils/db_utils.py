import psycopg as pg
from psycopg import sql
from pgvector.psycopg import register_vector
from logging import Logger

class PGConnector:
    """
    Configures a connection to a postgresql database with pgvector

    Attributes
    ----------
    _url: str
        The connection string used to connect to the database
    _logger: Logger

    _schema: str
        The schema 
    """
    def __init__(
            self,
            db_user: str,
            db_password: str,
            db_host: str,
            db_port: int,
            db_name: str,
            db_schema: str,
            logger: Logger,
            ) -> None:
        self._url: str = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        self._logger = logger
        self._schema = db_schema

    def check_extension(self):
        """
        Check whether the connected database has the pgvector extension installed
        """
        with pg.connect(self._url) as conn:
            self._logger.info("Connected to database")
            try:
                conn.execute(sql.SQL("""
                             CREATE EXTENSION IF NOT EXISTS vector;
                             """))
                self._logger.info("Vector extension is active")
            except pg.Error as e:
                raise e

    def reset_embedding_table(self, embedding_dimension: int) -> None:
        """
        Drop any existing embedding table, then create a new one with the right vector dimension
        """
        with pg.connect(self._url) as conn:
            register_vector(conn)
            with conn.cursor() as table_manage_cursor:
                table_manage_cursor.execute(
                sql.SQL("""
                DROP TABLE IF EXISTS {};
                """).format(sql.Identifier(self._schema, "embeddings"))
                )
                self._logger.info(f"Creating a table for {embedding_dimension} dimensional vectors")
                table_manage_cursor.execute(
                        sql.SQL("""
                        CREATE TABLE {} (
                            concept_id  int,
                            embedding  vector({})
                        );
                        """).format(sql.Identifier(self._schema, "embeddings"), embedding_dimension)
                        )
                conn.commit()
    
    def get_connection(self) -> pg.Connection:
        """
        Return a configured connection
        """
        conn = pg.connect(self._url)
        register_vector(conn)
        return conn
