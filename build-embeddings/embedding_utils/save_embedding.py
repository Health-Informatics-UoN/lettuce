import polars as pl
from pathlib import Path
from torch import Tensor
import psycopg as pg
from psycopg import sql

def save_parquet(path: Path, concepts: list[tuple[int, str, Tensor]]):
    pl.DataFrame(
            {
                "concept_id": [c[0] for c in concepts],
                "concept_name": [c[1] for c in concepts],
                "embeddings": [c[2] for c in concepts],
                }
            ).write_parquet(path)

def copy_to_postgres(cursor: pg.Cursor, concepts: list[tuple[int, str, Tensor]], db_schema: str, embedding_table_name: str) -> None:
    with cursor.copy(
            sql.SQL("COPY {} (concept_id, embedding) FROM STDIN WITH (FORMAT BINARY)").format(sql.Identifier(db_schema, embedding_table_name))
            ) as copy:
        copy.set_types(["int4", "vector"])
        for entry in concepts:
            copy.write_row((entry[0], entry[2]))
