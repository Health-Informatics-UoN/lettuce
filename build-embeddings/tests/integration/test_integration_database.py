import logging
import sys
from jinja2 import Environment
import pytest
import os
from psycopg import sql

from options.base_options import BaseOptions
from sentence_transformers import SentenceTransformer
from embedding_utils.concept_readers import PostgresConceptExtractor
from embedding_utils.db_utils import PGConnector
from embedding_utils.embedder import BatchEmbedder
from embedding_utils.fetch_concept_batches import BatchEmbeddingPipeline
from embedding_utils.save_embedding import PostgresWriter

pytestmark = pytest.mark.skipif(os.getenv('SKIP_DATABASE_TESTS') == 'true', reason="Skipping database tests")

logger = logging.Logger("embedding logger")
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def test_database_batch_pipeline():
    settings = BaseOptions()

    connector = PGConnector(
            db_user=settings.db_user,
            db_password=settings.db_password,
            db_host=settings.db_host,
            db_port=settings.db_port,
            db_name=settings.db_name,
            db_schema=settings.db_schema,
            logger=logger
            )

    connector.check_extension()
    connector.reset_embedding_table(embedding_dimension=384)
    
    with connector.get_connection() as conn:
        hopefully_empty = conn.execute(sql.SQL("SELECT * FROM {}").format(sql.Identifier(connector.db_schema, connector._embeddings_table_name))).fetchall()

        assert len(hopefully_empty) == 0
        

    reader = PostgresConceptExtractor(connector, batch_size=512, logger=logger)

    environment = Environment()
    template = environment.from_string("{{concept_name}}")
    embedder = BatchEmbedder(embedding_model=SentenceTransformer("BAAI/bge-small-en-v1.5"), template=template)

    store = PostgresWriter(connector)

    pipeline = BatchEmbeddingPipeline(
            reader=reader,
            embedder=embedder,
            store=store
            )

    pipeline.run_pipeline()

    with connector.get_connection() as conn:
        hopefully_not_empty = conn.execute(sql.SQL("SELECT * FROM {}").format(sql.Identifier(connector.db_schema, connector._embeddings_table_name))).fetchall()

        assert len(hopefully_not_empty) != 0

