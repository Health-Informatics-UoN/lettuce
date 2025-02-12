#!/bin/bash

for file in *.parquet
do
  uv run load_embeddings.py --embeddings-path $file
done
