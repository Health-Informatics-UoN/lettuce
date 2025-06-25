from os import getenv
from typing import List
from logging import Logger
from pgvector.psycopg import register_vector
import psycopg
from sentence_transformers import SentenceTransformer
#consider importing tqdm to estimate running of batches

DB_SCHEMA = getenv("DB_SCHEMA")

model = SentenceTransformer(getenv("EMBEDDING_MODEL"))

#TODO: use method to get embedding length
vector_length = "???"

logger = Logger("embeddings-upload")

uri = f"postgresql://{getenv('DB_USER')}:{getenv('DB_PASSWORD')}@{getenv('DB_HOST')}:{getenv('DB_PORT')}/{getenv('DB_NAME')}"

#TODO: type annotations
def embed_batch(cursor,
                concept_list: List[tuple[int,str]],
                embedding_model: SentenceTransformer,
                ):
    embeddings = embedding_model.encode([emb for _, emb in concept_list])
    with cursor.copy(
            f"COPY {DB_SCHEMA}.embeddings (concept_id, embedding) FROM STDIN WITH (FORMAT BINARY)"
            ) as copy:
        copy.set_types(["int4", "vector"])
        for entry in zip([id for id,_ in concept_list], embeddings):
            copy.write_row((entry[0], entry[1]))
    

with psycopg.connect(uri) as conn:
    logger.info("Connected to database\n")
    logger.info("Loading pgvector extension")

    conn.execute("""
                 CREATE EXTENSION vector;
                 """)
    register_vector(conn)
    print("Registered vector type")
    with conn.cursor() as cursor:
        cursor.execute(
                f"""
                DROP TABLE IF EXISTS {DB_SCHEMA}.embeddings;
                """
                )
        logger.info(f"Creating a table for {vector_length} dimensional vectors")
        cursor.execute(
                f"""
                CREATE TABLE cdm.embeddings (
                    concept_id  int,
                    embedding  vector({vector_length})
                );
                """
                )
        #TODO: figure out how to fetch concepts in batches, then use embed_batch to encode and copy them
        conn.commit()
