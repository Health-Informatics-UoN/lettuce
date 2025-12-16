import logging
from pathlib import Path
from typing import Literal, Annotated
from jinja2 import Environment
from sentence_transformers import SentenceTransformer
import torch
import typer
from options.base_options import BaseOptions

from embedding_utils.fetch_concept_batches import ConceptCsvEmbedder, PostgresConceptEmbedder

settings = BaseOptions()
logger = logging.Logger("embedding logger")
app = typer.Typer()

@app.command()
def embed_concepts(
        concept_source: Annotated[Literal["postgres", "csv"], typer.Option(help="The source to use for concepts")],
        embedding_model: Annotated[str, typer.Option(help="String to fetch a SentenceTransformer")] = "BAAI/bge-small-en-v1.5",
        template: Annotated[str, typer.Option(help="String specification for a Jinja2 template for rendering a concept")] = "{{concept_name}}",
        fetch_batch_size: Annotated[int, typer.Option(help="Number of concepts to extract at once if using the database")] = 16384,
        embed_batch_size: Annotated[int, typer.Option(help="Number of embeddings to generate at once if using the database")] = 512,
        db_load_method: Annotated[Literal["replace", "extend"], typer.Option(help="How to load embeddings in the database. If 'replace', drops any existing embeddings table. Otherwise extends the table")] = "extend",
        source_path: Annotated[str | None, typer.Option(help="Path for source csv if reading from file")] = None,
        save_method: Annotated[Literal["load_to_database", "save_to_parquet"], typer.Option(help="Whether to save the embeddings to a file or load them into your database (only if loading from a database)")] = "save_to_parquet",
        output_path: Annotated[str, typer.Option(help="Path to save a parquet file")] = "embeddings.parquet",
        ):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else None
    logger.info(f"Using device '{device}' for embeddings")
    template_env = Environment()
    concept_template = template_env.from_string(template)
    if concept_source == "postgres":
        embedder = PostgresConceptEmbedder(
                db_user=settings.db_user,
                db_password=settings.db_password,
                db_host=settings.db_host,
                db_port=settings.db_port,
                db_name=settings.db_name,
                db_schema=settings.db_schema,
                template=concept_template,
                embedding_model=SentenceTransformer(embedding_model),
                logger=logger,
                fetch_batch_size=fetch_batch_size,
                embed_batch_size=embed_batch_size,
                )
        embedder.check_extension()
        if db_load_method == "replace":
            embedder.reset_embedding_table()
        if save_method == "load_to_database":
            embedder.load_embeddings()
        elif save_method == "save_to_parquet":
            save_path = Path(output_path)
            embedder.save_embeddings_to_parquet(save_path)
    elif concept_source == "csv":
        embedder = ConceptCsvEmbedder(
                embedding_model = SentenceTransformer(embedding_model),
                template=concept_template,
                logger=logger,
                )
        embedder.load_concepts(Path(source_path))
        embedder.embed_concepts()
        embedder.save_embeddings_to_parquet(Path(output_path))



if __name__ == "__main__":
    typer.run(embed_concepts)
