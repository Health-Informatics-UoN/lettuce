from logging import Logger
from pathlib import Path
from jinja2 import Template
from torch import Tensor
import psycopg as pg
from psycopg import sql
import polars as pl
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from embedding_utils.string_building import Concept
from embedding_utils.save_embedding import copy_to_postgres

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
            logger: Logger,
            fetch_batch_size: int = 16384,
            embed_batch_size: int = 512,
            ) -> None:
        self._url: str = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        self._template: Template = template
        self._logger: Logger = logger
        self._schema: str = db_schema
        self._embedding_model: SentenceTransformer = embedding_model
        self._embed_batch_size = embed_batch_size
        self._fetch_batch_size = fetch_batch_size
        self._concept_query = sql.SQL("SELECT concept_id, concept_name, domain_id, vocabulary_id, concept_class_id FROM {}").format(sql.Identifier(db_schema, "concept"))
        self._log_example = True

    def check_extension(self):
        with pg.connect(self._url) as conn:
            self._logger.info("Connected to database")
            try:
                conn.execute(sql.SQL("""
                             CREATE EXTENSION IF NOT EXISTS vector;
                             """))
                self._logger.info("Vector extension is active")
            except pg.Error as e:
                raise e

    def reset_embedding_table(self):
        with pg.connect(self._url) as conn:
            register_vector(conn)
            with conn.cursor() as table_manage_cursor:
                table_manage_cursor.execute(
                sql.SQL("""
                DROP TABLE IF EXISTS {};
                """).format(sql.Identifier(self._schema, "embeddings"))
                )
                self._logger.info(f"Creating a table for {self._embedding_model.get_sentence_embedding_dimension()} dimensional vectors")
                table_manage_cursor.execute(
                        sql.SQL("""
                        CREATE TABLE {} (
                            concept_id  int,
                            embedding  vector({})
                        );
                        """).format(sql.Identifier(self._schema, "embeddings"), self._embedding_model.get_sentence_embedding_dimension())
                        )
                conn.commit()

    def embed_batch(self, concept_batch: list[Concept]) -> list[tuple[int, str, Tensor]]:
        concept_strings = [c.render_concept_as_template(self._template) for c in concept_batch]
        if self._log_example:
            self._logger.info(f"Example concept string: {concept_strings [0]}")
            self._log_example = False
        concept_embeddings = self._embedding_model.encode([c[2] for c in concept_strings], convert_to_tensor=False, show_progress_bar=True).tolist()
        return list(zip([id for id,_,_ in concept_strings], [name for _, name,_ in concept_strings], concept_embeddings))

    def load_embeddings(self):
        with pg.connect(self._url) as conn:
            with conn.cursor("concept cursor") as concept_cursor:
                concept_cursor.itersize = self._fetch_batch_size
                concept_cursor.execute(self._concept_query)

                with conn.cursor("embed cursor") as embed_cursor:
                    while True:
                        rows = concept_cursor.fetchmany(self._embed_batch_size)
                        if len(rows) > 0:
                            concepts = [
                                    Concept(
                                        concept_id=r[0],
                                        concept_name=r[1],
                                        domain=r[2],
                                        vocabulary=r[3],
                                        concept_class=r[4],
                                        ) for r in rows
                                    ]
                            copy_to_postgres(embed_cursor, self.embed_batch(concepts), self._schema, "embeddings")
                        else:
                            break
                    conn.commit()

    def save_embeddings_to_parquet(self, path: Path):
        dfs = []
        with pg.connect(self._url) as conn:
            with conn.cursor("concept cursor") as concept_cursor:
                concept_cursor.itersize = self._fetch_batch_size
                concept_cursor.execute(self._concept_query)

                while True:
                    rows = concept_cursor.fetchmany(self._embed_batch_size)
                    if len(rows) > 0:
                        concepts = [
                                Concept(
                                    concept_id=r[0],
                                    concept_name=r[1],
                                    domain=r[2],
                                    vocabulary=r[3],
                                    concept_class=r[4],
                                    ) for r in rows
                                ]
                        embeddings = self.embed_batch(concepts)
                        dfs.append(
                                pl.DataFrame(embeddings, schema={
                                    "concept_id": pl.Int32(),
                                    "concept_name": pl.String(),
                                    "embedding": pl.Array(pl.Float64, self._embedding_model.get_sentence_embedding_dimension())}, orient="row"
                                             )
                                )
                    else:
                        break
        pl.concat(dfs).write_parquet(path)



class ConceptCsvEmbedder():
    def __init__(
            self,
            embedding_model: SentenceTransformer,
            template: Template,
            logger: Logger,
            table_schema: dict[str, pl.DataType] = CONCEPT_SCHEMA
            ) -> None:
        self._embedding_model = embedding_model
        self._template = template
        self._logger = logger
        self._table_schema = table_schema

    def load_concepts(self, path: Path) -> None:
        if hasattr(self, "_concept_df"):
            self._concept_df.extend(pl.read_csv(path, schema=self._table_schema, separator="\t", quote_char=None))
        else:
            self._concept_df = pl.read_csv(path, schema=self._table_schema, separator="\t", quote_char=None)

    def embed_concepts(self):
        concept_strings = [
        self._template.render({
            "concept_name": row["concept_name"],
            "domain": row["domain_id"],
            "vocabulary": row["vocabulary_id"],
            "concept_class": row["concept_class_id"]
            }) for row in self._concept_df.select([
                "concept_name", "domain_id", "vocabulary_id", "concept_class_id"
                ]).iter_rows(named=True)
            ]
        self._logger.info(f"Example concept string: {concept_strings[0]}")

        embeddings = self._embedding_model.encode(
                concept_strings,
                convert_to_tensor=False,
                show_progress_bar=True,
                )

        self._concept_df = self._concept_df.with_columns(
            pl.Series(
                "embeddings",
                embeddings.tolist(),
                dtype=pl.Array(
                    pl.Float64,
                    self._embedding_model.get_sentence_embedding_dimension()
                    )
                )
        )
        self._logger.info(f"Embedded {len(self._concept_df)} concepts")
    
    def save_embeddings_to_parquet(self, path: Path):
        self._concept_df.select(["concept_id", "concept_name", "embeddings"]).write_parquet(path)
        self._logger.info(f"Wrote embeddings to {path}")
