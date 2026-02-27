import pytest
import polars as pl

from embedding_utils.protocols import EmbeddedConcept
from embedding_utils.save_embedding import ParquetWriter

def test_parquet_writer(tmp_path):
    concepts = [
            EmbeddedConcept(
                concept_id=345678,
                concept_name="This isn't a real concept",
                embedding=[1,2,3,4,5]
                ),
            EmbeddedConcept(
                concept_id=6789876,
                concept_name="Surprise, surprise, neither is this one",
                embedding=[2,3,4,5,6]
                )
            ]
    writer = ParquetWriter(tmp_path / "test.parquet")

    writer.save(concepts)
    writer.save(concepts)

    read_output = pl.read_parquet(tmp_path / "test.parquet")

    assert read_output.columns == ["timestamp", "concept_id", "concept_name", "embeddings"]
    assert len(read_output) == 4
    assert len(read_output.filter(pl.col("concept_id") == 345678)) == 2
