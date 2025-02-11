This directory contains code for loading embeddings from a parquet file to a ChromaDB collection.
It was made so that a vector database that can run locally could be shared among hackathon participants.

The `load_embeddings.py` script will connect to a persistent ChromaDB, which is a sqlite3 file, or create one if that doesn't exist.
It's written to run with inline dependencies, but if you can't do that, you can read the start of the script to install them yourself.
For me the easiest way is with `uv run`.
Configuration is through a `.env` file, which you can pass to uv with


```
uv run --env-file .env load_embeddings.py
```

Here's mine:

```
CHROMA_PATH=chroma_db
COLLECTION_NAME=omop
DISTANCE_METRIC=cosine
BATCH_SIZE=80000
EMBEDDINGS_PATH=embeddings.parquet
```

I had the `BATCH_SIZE` parameter in there from code adding items to other vector databases where the number of embeddings loaded into memory was an issue.
ChromaDB will only load 83333 at a time, so unless you have enormous embeddings, this won't be an issue, and you can just leave it there.
Loading ~9.6 million embeddings this way took about 3 hours. Non-trivial, but not terrible.
