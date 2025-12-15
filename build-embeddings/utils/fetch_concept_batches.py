from logging import Logger
from typing import Literal
from jinja2 import Template
import psycopg as pg
from psycopg import sql
import polars as pl
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

class PostgresConceptEmbedder():
    def __init__(
            self,
            db_user: str,
            db_password: str,
            db_host: str,
            db_port: int,
            db_name: str,
            db_schema: str,
            template: Template,
            embedding_model: SentenceTransformer,
            save_method: Literal["copy", "parquet"],
            logger: Logger,
            ) -> None:
        self._url: str = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        self._template: Template = template
        self._save_method: Literal["copy", "parquet"] = save_method
        self._logger: Logger = logger

    def embed_concepts(self):
        with pg.connect(self._url) as conn:
            self._logger.info("Connected to database")
            try:
                conn.execute(sql.SQL("""
                             CREATE EXTENSION IF NOT EXISTS vector;
                             """))
            except pg.Error as e:
                raise e

        
