from os import environ
from typing import List
import logging
from pgvector.psycopg import register_vector
import psycopg
from psycopg import sql
from sentence_transformers import SentenceTransformer
import sys
#consider importing tqdm to estimate running of batches


try:
    DB_SCHEMA = environ["DB_SCHEMA"]
    DB_USER = environ['DB_USER']
    DB_PASSWORD = environ['DB_PASSWORD']
    DB_HOST = environ['DB_HOST']
    DB_PORT = environ['DB_PORT']
    DB_NAME = environ['DB_NAME']
except KeyError:
    sys.exit("Couldn't read database arguments")

try:
    model = SentenceTransformer(environ["EMBEDDING_MODEL"])
except KeyError:
    sys.exit("Couldn't read the embedding model")

vector_length = model.get_sentence_embedding_dimension()

logger = logging.getLogger(__name__)
logging.basicConfig(filename='embedding_load.log', level=logging.INFO)

uri = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
logger.info(f"Connecting to database at {uri}")
print(f"Connecting to database at {uri}")

def embed_batch(cursor: psycopg.Cursor,
                concept_list: List[tuple[int,str]],
                embedding_model: SentenceTransformer,
                ) -> None:
    embeddings = embedding_model.encode([emb for _, emb in concept_list], batch_size=128)
    with cursor.copy(
            sql.SQL("COPY {} (concept_id, embedding) FROM STDIN WITH (FORMAT BINARY)").format(sql.Identifier(DB_SCHEMA, "embeddings"))
            ) as copy:
        copy.set_types(["int4", "vector"])
        for entry in zip([id for id,_ in concept_list], embeddings):
            copy.write_row((entry[0], entry[1]))
    

with psycopg.connect(uri) as conn:
    logger.info("Connected to database\n")
    logger.info("Loading pgvector extension")

    # conn.execute("""
    #              CREATE EXTENSION IF NOT EXISTS vector;
    #              """)
    register_vector(conn)
    print("Registered vector type")
    with conn.cursor() as table_manage_cursor:
        table_manage_cursor.execute(
                sql.SQL("""
                DROP TABLE IF EXISTS {};
                """).format(sql.Identifier(DB_SCHEMA, "embeddings"))
                )
        logger.info(f"Creating a table for {vector_length} dimensional vectors")
        table_manage_cursor.execute(
                sql.SQL("""
                CREATE TABLE {} (
                    concept_id  int,
                    embedding  vector({})
                );
                """).format(sql.Identifier(DB_SCHEMA, "embeddings"), vector_length)
                )
        conn.commit()
    with conn.cursor(name="concept_fetch") as concept_cursor:
        concept_cursor.itersize = 16384

        query = sql.SQL("SELECT concept_id, concept_name FROM {}").format(sql.Identifier(DB_SCHEMA, "concept"))

        concept_cursor.execute(query)

        with conn.cursor() as embed_cursor:
            while True:
                rows = concept_cursor.fetchmany(512)
                if len(rows) > 0:
                    embed_batch(embed_cursor, rows, model)
                else:
                    break
            conn.commit()
