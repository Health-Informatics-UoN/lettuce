# Build embeddings

Lettuce requires a table of embeddings to read from for semantic search.
If you have a parquet file of embeddings for your vocabularies, you can load them into a new postgres OMOP-CDM database as configured in [omop-lite](https://github.com/health-informatics-uon/omop-lite).
This module lets you generate embeddings from a table of vocabularies, and either load these into a postgres database, or write them to a parquet file.
If you want to load the embeddings into a postgres database, it must have [PGVector](https://github.com/pgvector/pgvector) installed.
The vocabularies can be extracted from either the postgres database you want to load embeddings into, or a tab-delimited file, as downloaded from [Athena](https://athena.ohdsi.org/search-terms/start).

The embeddings can use attributes of each concept using [Jinja2](https://jinja.palletsprojects.com/en/stable/) templates.
The default is just to use the concept name.
A simple example of what's possible is:

| Template | Example result |
|----------|----------------|
| `{{concept_name}}` | Conjunctival concretion |
| `{{concept_name}}, a {{concept_class}} {{domain}}` | Conjunctival concretion, a Disorder Condition |

## Usage

If you install `build-embeddings`, it can be run with that command.
Otherwise, you can run it with `uv run build-embeddings [ARGS]`

| Argument | Type | Description |
|----------|------|-------------|
| --concept-source | postgres/csv | The source to use for concepts [required] |
| --embedding-model | TEXT | String to fetch a SentenceTransformer [default: BAAI/bge-small-en-v1.5] |
| --template | TEXT | String specification for a Jinja2 template for rendering a concept [default: {{concept_name}}] |
| --fetch-batch-size | INTEGER | Number of concepts to extract at once if using the database [default: 16384] |
| --embed-batch-size | INTEGER | Number of embeddings to generate at once if using the database [default: 512] |
| --db-load-method | replace/extend | How to load embeddings in the database. If 'replace', drops any existing embeddings table. Otherwise extends the table [default: extend] |
| --source-path | TEXT | Path for source csv if reading from file |
| --save-method | load_to_database/save_to_parquet | Whether to save the embeddings to a file or load them into your database (only if loading from a database) [default: save_to_parquet] |
| --output-path | TEXT | If saving to a parquet file, the path for output |

You can show these arguments and their descriptions with `uv run build-embeddings --help`

To write a parquet file from a vocabulary csv:

```bash
uv run build-embeddings --concept-source csv --source-path VOCABULARY.csv --output-path embeddings.parquet
```
