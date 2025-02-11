import os
import requests
from argparse import ArgumentParser
import time

from qdrant_client import QdrantClient

parser = ArgumentParser()

parser.add_argument("-u", "--url", type=str, default="http://localhost:6333")
parser.add_argument("-c", "--collection-id", type=str, default="omop")
parser.add_argument("-d", "--snapshot-directory", type=str, default="qdrant-snapshots")
parser.add_argument("-a", "--attempts", type=int, default=10)
parser.add_argument("-i", "--delay-interval", type=int, default=60)

args = parser.parse_args()

client = QdrantClient(args.url, timeout=3600)
print("Creating snapshot")
client.create_snapshot(collection_name=args.collection_id)
print(client.list_snapshots(args.collection_id))
snapshot_url = (
    f"{args.url}/collections/{args.collection_id}/snapshots/{snapshot_info.name}"
)


os.makedirs(args.snapshot_directory, exist_ok=True)

print("Creating local directory")
snapshot_name = os.path.basename(snapshot_url)
local_snapshot_path = os.path.join(args.snapshot_directory, snapshot_name)

response = requests.get(
    snapshot_url,
)

print("Downloading snapshot")
with open(local_snapshot_path, "wb") as f:
    for i in range(args.attempts):
        try:
            response.raise_for_status()
            f.write(response.content)
            break
        except requests.exceptions.Timeout:
            print(f"Failed to connect (attempt {i} of {args.attempts})")
            time.sleep(args.delay_interval)
