import logging
import sys
from jinja2 import Environment
from sentence_transformers import SentenceTransformer
import polars as pl
from pathlib import Path

from embedding_utils.concept_readers import CsvConceptExtractor
from embedding_utils.embedder import BatchEmbedder
from embedding_utils.fetch_concept_batches import BatchEmbeddingPipeline
from embedding_utils.save_embedding import ParquetWriter

logger = logging.Logger("embedding logger")
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def test_csv_batch_pipeline(tmp_path):
    reader = CsvConceptExtractor(path=Path("tests/test_data/CONCEPT.csv"), batch_size=512)

    environment = Environment()
    template= environment.from_string("{{concept_name}}")
    embedder = BatchEmbedder(embedding_model=SentenceTransformer("BAAI/bge-small-en-v1.5"), template=template)

    store = ParquetWriter(path=tmp_path / "embeddings.parquet")

    pipeline = BatchEmbeddingPipeline(
            reader=reader,
            embedder=embedder,
            store=store,
            )

    pipeline.run_pipeline()

    output = pl.read_parquet(tmp_path / "embeddings.parquet")

    assert len(output) == 1526
    assert output.columns == ["timestamp", "concept_id", "concept_name", "embeddings"]
