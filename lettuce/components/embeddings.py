from haystack_integrations.components.embedders.fastembed import (
    FastembedTextEmbedder,
)
from haystack import component
from haystack.dataclasses import Document
from typing import Any, List, Dict
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from omop.omop_queries import query_vector
from omop.db_manager import db_session

from options.base_options import BaseOptions
from options.pipeline_options import EmbeddingModelName, EmbeddingModel, EMBEDDING_MODELS

settings = BaseOptions()

# -------- Embedding Models -------- >


# I know there's a pgvector integration with haystack, but it seems to be able to only query a single table at a time
# https://github.com/deepset-ai/haystack-core-integrations/tree/main/integrations/pgvector/examples
@component
class PGVectorQuery:
    """
    A haystack component for retrieving concept information using embeddings in a postgres database with pgvector
    """
    def __init__(
            self,
            connection: Session,
            embed_vocab: List[str] | None = None,
            domain_id: List[str] | None = None,
            standard_concept:bool = False,
            valid_concept:bool = False,
            top_k: int = 5,
            ) -> None:
        self._connection = connection
        self._embed_vocab = embed_vocab
        self._domain_id = domain_id
        self._standard_concept = standard_concept
        self._valid_concept = valid_concept
        self._top_k = top_k

    @component.output_types(documents=List[Document])
    def run(
            self,
            query_embedding: List[float],
            describe_concept: bool = False,
            ):
        # only have cosine_similarity at the moment
        #TODO add selection of distance metric to query_vector
        query = query_vector(
                query_embedding=query_embedding,
                embed_vocab=self._embed_vocab,
                domain_id=self._domain_id,
                standard_concept=self._standard_concept,
                valid_concept=self._valid_concept,
                n = self._top_k,
                describe_concept=describe_concept,
                ) 
        try:
            query_results = self._connection.execute(query).mappings().all()
        except SQLAlchemyError as e:
            raise SQLAlchemyError(f"Vector query execution failed: {str(e)}")
        if describe_concept:
            return query_results
        else:
            try:
                return {"documents": [
                    Document(
                        id=res["id"],
                        content=res["content"],
                        score=res["score"],
                        ) for res in query_results]
                    }
            except KeyError as e:
                raise KeyError(f"Missing required key in query results: {str(e)}")

def get_embedding_model(name: EmbeddingModelName) -> EmbeddingModel:
    """
    Collects the details of an embedding model when given its name


    Parameters
    ----------
    name: EmbeddingModelName
        The name of an embedding model we have the details for

    Returns
    -------
    EmbeddingModel
        An EmbeddingModel object containing the name and the details used
    """
    return EmbeddingModel(name=name, info=EMBEDDING_MODELS[name])


class Embeddings:
    """
    This class allows the building or loading of a vector
    database of concept names. This database can then
    be used for vector search.

    Methods
    -------
    search:
        Query the attached embeddings database with provided
        search terms.
    """

    def __init__(
        self,
        model_name: EmbeddingModelName,
        embed_vocab: List[str] | None=None,
        domain_id: List[str] | None = None,
        standard_concept: bool=False,
        valid_concept: bool = False,
        top_k: int=5,
    ) -> None:
        """
        Initialises the connection to an embeddings database

        Parameters
        ----------
        embeddings_path: str
            A path for the embeddings database. If one is not found,
            it will be built, which takes a long time. This is built
            from concepts fetched from the OMOP database.

        force_rebuild: bool
            If true, the embeddings database will be rebuilt.

        embed_vocab: List[str]
            A list of OMOP vocabulary_ids. If the embeddings database is
            built, these will be the vocabularies used in the OMOP query.

        model: EmbeddingModel
            The model used to create embeddings.

        search_kwargs: dict
            kwargs for vector search.
        """
        self._model = get_embedding_model(model_name)
        self._embed_vocab = embed_vocab
        self._domain_id = domain_id
        self._standard_concept = standard_concept
        self._valid_concept = valid_concept
        self._top_k = top_k


    def get_embedder(self) -> FastembedTextEmbedder:
        """
        Get an embedder for queries in LLM pipelines

        Returns
        _______
        FastembedTextEmbedder
        """
        query_embedder = FastembedTextEmbedder(model=self._model.info.path, parallel=0)
        query_embedder.warm_up()
        return query_embedder

    def get_retriever(self) -> PGVectorQuery:
        """
        Get a retriever for LLM pipelines

        Returns
        -------
        PGVectorQuery
        """
        try:
            assert(self._model.info.dimensions == settings.db_vecsize)
            return PGVectorQuery(
                    db_session(),
                    embed_vocab=self._embed_vocab,
                    domain_id=self._domain_id,
                    standard_concept=self._standard_concept,
                    valid_concept=self._valid_concept,
                    top_k=self._top_k,
                    )
        except AssertionError:
            raise AssertionError(f"Embedder dimensions {str(self._model.info.dimensions)} not equal to vector store dimensions {str()}")

    def search(self, query: List[str]) -> List[List[Dict[str, Any]]]:
        """
        Search the attached vector database with a list of informal medications

        Parameters
        ----------
        query: List[str]
            A list of informal medication names

        Returns
        -------
        List[List[Dict[str, Any]]]
            For each medication in the query, the result of searching the vector database
        """
        retriever = self.get_retriever()
        query_embedder = FastembedTextEmbedder(
            model=self._model.info.path, parallel=0, prefix="query:"
        )
        query_embedder.warm_up()
        query_embeddings = [query_embedder.run(name) for name in query]
        result = [
            retriever.run(query_embedding["embedding"])
            for query_embedding in query_embeddings
        ]
        return [
            [
                {
                    "concept_id": doc.id,
                    "concept": doc.content,
                    "score": doc.score,
                }
                for doc in res["documents"]
            ]
            for res in result
        ]
