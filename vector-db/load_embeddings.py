from os import getenv
import pyarrow.parquet as pq
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, OptimizersConfigDiff
import numpy as np

uri = f"http://{getenv('VDB_HOST')}:{getenv('VDB_PORT')}"

print("Connecting to Qdrant")
client = QdrantClient(uri)

print("Opening embeddings")
parquet_file = pq.ParquetFile("embeddings/embeddings.parquet")

print("Getting vector length")
vector_length = len(next(parquet_file.iter_batches(batch_size=1))["embeddings"][0])

distance_metrics = {
    "cosine": Distance.COSINE,
    "dot": Distance.DOT,
}

print("Creating collection")
if not client.collection_exists(getenv("COLLECTION_NAME")):
    client.create_collection(
        collection_name=getenv("COLLECTION_NAME"),
        vectors_config=VectorParams(
            size=vector_length, distance=distance_metrics[getenv("DISTANCE_METRIC")]
        ),
        optimizers_config=OptimizersConfigDiff(indexing_threshold=0),
    )


print("Uploading vectors")
for batch in tqdm(parquet_file.iter_batches(batch_size=400_000)):
    vectors = batch["embeddings"].to_numpy(zero_copy_only=False)
    payload = [
        {"concept_name": str(concept_name), "concept_id": str(concept_id)}
        for concept_name, concept_id in zip(batch["concept_name"], batch["concept_id"])
    ]

    client.upload_collection(
        collection_name=getenv("COLLECTION_NAME"),
        vectors=vectors,
        payload=payload,
        ids=None,
        batch_size=256,
    )

print("Starting indexing")
client.update_collection(
    collection_name=getenv("COLLECTION_NAME"),
    optimizers_config=OptimizersConfigDiff(indexing_threshold=20_000),
)
