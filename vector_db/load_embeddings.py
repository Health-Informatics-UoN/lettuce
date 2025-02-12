# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "chromadb",
#     "numpy",
#     "pyarrow",
#     "tqdm",
# ]
# ///
from argparse import ArgumentParser

import chromadb
import numpy as np
from tqdm import tqdm
import pyarrow.parquet as pq

parser = ArgumentParser()
parser.add_argument("--chroma-path", type=str, default="chroma_db")
parser.add_argument("--distance-metric", type=str, default="cosine")
parser.add_argument("--collection-name", type=str, default="omop")
parser.add_argument("--embeddings-path", type=str)
parser.add_argument("--batch-size", type=int, default=80_000)

args = parser.parse_args()

chroma_client = chromadb.PersistentClient(args.chroma_path)

try:
    collection = chroma_client.get_collection(args.collection_name)
    print("Collection already present")
except chromadb.errors.InvalidCollectionException:
    collection = chroma_client.create_collection(
        args.collection_name,
        metadata={"hnsw:space": args.distance_metric},
    )

print("Opening embeddings")
parquet_file = pq.ParquetFile(args.embeddings_path)

print("Uploading vectors")
for batch in tqdm(parquet_file.iter_batches(batch_size=args.batch_size)):
    vectors = [
        emb.astype(np.float32)
        for emb in batch["embeddings"].to_numpy(zero_copy_only=False)
    ]
    ids = [str(concept_id) for concept_id in batch["concept_id"]]

    collection.add(embeddings=vectors, ids=ids)
