# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "chromadb",
#     "numpy",
#     "pyarrow",
#     "tqdm",
# ]
# ///
from os import getenv

import chromadb
import numpy as np
from tqdm import tqdm
import pyarrow.parquet as pq

chroma_client = chromadb.PersistentClient(getenv("CHROMA_PATH"))
collection = chroma_client.create_collection(
    getenv("COLLECTION_NAME"),
    metadata={"hnsw:space": getenv("DISTANCE_METRIC")},
)

print("Opening embeddings")
parquet_file = pq.ParquetFile(getenv("EMBEDDINGS_PATH"))

print("Uploading vectors")
for batch in tqdm(parquet_file.iter_batches(batch_size=int(getenv("BATCH_SIZE")))):
    vectors = [
        emb.astype(np.float32)
        for emb in batch["embeddings"].to_numpy(zero_copy_only=False)
    ]
    ids = [str(concept_id) for concept_id in batch["concept_id"]]

    collection.add(embeddings=vectors, ids=ids)
