from enum import Enum
from urllib.parse import quote_plus
from dotenv import load_dotenv
from haystack.dataclasses import Document
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.components.embedders.fastembed import (
    FastembedDocumentEmbedder,
    FastembedTextEmbedder,
)
import os
from os import environ
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from typing import Any, List, Dict
from pydantic import BaseModel

from omop.omop_models import Concept

# ---------------------------------------------------------- >
# ------ Information about Embedding Models ------ >


# ------ Bidirectional Gated Encoder ------>
# ------ { BGE-small } ------- >

"""
BGESMALL
--------
    - Model Name: Bidirectional Gated Encoder
    
    - Version: Small
    
    - Dimensions: 384
    
    - Description:
        BGE-small has been designed for tasks such as sentence
        embedding, semantic similarity, and information retrieval. 
        
    - Benifits:
        BGE-small more efficient in terms of speed and memory,
        while still being capable of generating high-quality
        sentence or document embeddings.
        
    - Research Paper:
        arXiv:2402.03216 
"""


# ------ SBERT (Sentence-BERT) ------- >
# ------ MiniLM { MINILM } ------- >

"""
MINILM
------
    - Model Name: Sentence-BERT
    
    - Version: MiniLM
    
    - Dimensions: 384
    
    - Description:
        MiniLM is a smaller, more efficient version of the 
        Sentence-BERT (SBERT) model, specifically optimized 
        for tasks such as semantic similarity, sentence 
        embeddings, and information retrieval.
        
    - Benifits:
        The MiniLM variant provides a good trade-off between 
        computational efficiency and performance, making it
        suitable for use cases where speed and resource 
        limitations are important, without sacrificing 
        too much accuracy.
        
    - Research Paper:
        arXiv:1908.10084
"""

# ------ Generalizable T5 Retrieval ------- >
# ------ Base { GTR_T5_BASE } ------- >

"""
GTR_T5_BASE
-----------
    - Model Name: Generalizable T5 Retrieval
    
    - Version: Base
    
    - Dimensions: 768
    
    - Description:
        GTR-T5 is a dense retrieval model using a dual encoder 
        architecture for efficient semantic search and passage 
        retrieval. It encodes queries and documents separately
        into a shared embedding space, allowing fast and scalable
        retrieval using a dot-product similarity. This model 
        is based on T5 and optimized for generalization 
        across diverse tasks.

    - Benifits:
        GTR-T5 is highly generalizable across different tasks, scalable
        in size, and efficient for large-scale retrieval with precomputed 
        document embeddings. 
        
        It leverages the T5 model for deep semantic understanding and 
        ensures fast retrieval using Approximate Nearest Neighbor 
        (ANN) search, making it both powerful and efficient 
        for various retrieval tasks.
        
    - Research Paper:
        ArXiv:abs/2112.07899

"""

# ------ Generalizable T5 Retrieval ------- >
# ------ Large { GTR_T5_LARGE } ------- >

"""
GTR_T5_LARGE
------------
    - Model Name: Generalizable T5 Retrieval
    
    - Version: Large
    
    - Dimensions: 1024
    
    - Description:
        GTR-T5-Large is a powerful version of the Generalizable T5 
        Retrieval model designed for dense retrieval tasks. 
        
        It encodes queries and documents into a shared embedding 
        space to enable efficient retrieval. The large version 
        enhances performance by offering more capacity for 
        complex semantic understanding.
        
    - Benifits:
        GTR-T5-Large excels at generalizing across various tasks and domains.
        It is scalable and ideal for handling large datasets while 
        maintaining efficient retrieval. Leveraging T5’s deep 
        language understanding, it supports fast retrieval with 
        Approximate Nearest Neighbor (ANN) search, making it 
        highly effective for large-scale semantic search 
        and retrieval tasks.
        
    - Research Paper:
        ArXiv:abs/2112.07899
"""

# ------ Embedding Models for Search Engines ------- >
# ------ Base { E5_BASE } ------- >

"""
E5_BASE
-------
    - Model Name: Embedding Models for Search Engines
    
    - Version: Base
    
    - Dimensions: 768
    
    - Description:
        E5 is a family of dense retrieval models by Microsoft, designed
        to generate high-quality text embeddings for search and 
        retrieval tasks. 
        
        It leverages contrastive learning on multilingual text pairs,
        combined with supervised fine-tuning, to perform well in 
        zero-shot and fine-tuned settings. 
        
        The base version provides efficient embeddings for tasks
        like semantic search, passage retrieval, document 
        ranking, and clustering.

    - Benifits:
        E5 offers high-quality semantic embeddings that generalize well across
        different domains and tasks. 
        
        Fine-tuned on the BEIR benchmark, it excels in cross-domain 
        retrieval and semantic search scenarios. 
        
        E5 also supports instruction-tuned variants for enhanced 
        task-specific performance, and demonstrates strong
        results on retrieval benchmarks like BEIR and MTEB.
        
    - Research Paper:
        arXiv:2212.03533

"""

# ------ Embedding Models for Search Engines ------- >
# ------ Large { E5_LARGE } ------- >

"""
E5_LARGE
--------
    - Model Name: Embedding Models for Search Engines
    
    - Version: Large
    
    - Dimensions: 1024
    
    - Description:
        E5-Large is an advanced version of Microsoft’s E5 family of 
        dense retrieval models, designed for generating high-quality 
        text embeddings for search, retrieval, and ranking tasks. 
        
        Like E5-Base, it utilizes contrastive learning with multilingual
        text pairs and fine-tuning on supervised datasets, 
        but the large version offers greater capacity, 
        improving performance on more complex tasks.
        
    - Benifits:
        E5-Large provides deeper semantic understanding due to its 
        larger model size, offering improved performance on 
        retrieval tasks across diverse domains. 
        
        It excels in semantic search, cross-domain retrieval,
        and document ranking, leveraging its larger capacity
        for better generalization and accuracy. 
        
        E5-Large demonstrates strong results on benchmarks 
        such as BEIR and MTEB.
        
    - Research Paper:
        arXiv:2212.03533

"""

# ------ DistilBERT ------- >
# ------ Base Uncased { DISTILBERT_BASE_UNCASED } ------- >

"""
DISTILBERT_BASE_UNCASED
-----------------------
    - Model Name: DistilBERT
    
    - Version: Base Uncased
    
    - Dimensions: 768
    
    - Description:
        DistilBERT is a smaller, faster, and lighter version of 
        the BERT model designed by Hugging Face for NLP tasks. 
        
        It offers 97% of BERT's performance but is 40% smaller, 
        making it ideal for deployment in resource-constrained
        environments. 
        
        DistilBERT reduces computational overhead, enabling faster
        inference while retaining high accuracy on most tasks.
        
    - Benifits:
        DistilBERT is 40% smaller and 60% faster than BERT, requiring 
        fewer resources while retaining 97% of BERT’s performance, 
        making it ideal for efficient deployment and easy 
        fine-tuning in resource-constrained environments.
        
    - Research Paper:
        arXiv:1910.01108
    
"""

# ------ distiluse-base-multilingual-cased-v1 ------- >
# ------ Base Multilingual { DISTILUSE_BASE_MULTILINGUAL } ------- >

"""
DISTILUSE_BASE_MULTILINGUAL
---------------------------
    - Model Name: distiluse-base-multilingual-cased-v1
    
    - Version: Base Multilingual
    
    - Dimensions: 512
    
    - Description:
        DistilUSE is a multilingual variant of the DistilBERT model 
        by Hugging Face, optimized for generating high-quality 
        multilingual text embeddings. 
        
        It is pre-trained on a large-scale multilingual corpus, 
        enabling it to encode text from multiple languages 
        into a shared embedding space.
        
    - Benifits:
        DistilUSE offers high-quality multilingual embeddings that 
        generalize well across different languages and tasks. 
        
        It is efficient for cross-lingual search, semantic similarity, 
        and retrieval tasks, making it ideal for multilingual 
        applications and scenarios.
        
    - Research Paper:
        ArXiv. /abs/1910.01108

"""

# ------ Contriever ------- >
# ------ Contriever { CONTRIEVER } ------- >

"""
CONTRIEVER
----------
    - Model Name: Contriever
    
    - Version: Contriever

    - Dimensions: 768
    
    - Description:
        Contriever, developed by Facebook, is an unsupervised dense 
        retrieval model designed for semantic search and information
        retrieval tasks without the need for labeled data.
        
        Using contrastive learning, it generates high-quality text
        embeddings for tasks like zero-shot retrieval, making it
        effective in domains where no task-specific 
        data is available. 
        
    - Benifits:
        Contriever excels at unsupervised dense retrieval, offering
        strong zero-shot performance across various domains 
        using contrastive learning, and is highly versatile,
        achieving good results in fields like biomedical,
        legal, and scientific datasets without 
        task-specific supervision.
        
    - Research Paper:
        ArXiv. /abs/2112.09118
"""

# -------- Embedding Models -------- >


class EmbeddingModelName(str, Enum):
    """
    This enumerates the embedding models we have the download details for
    """

    BGESMALL = "BGESMALL"
    MINILM = "MINILM"
    GTR_T5_BASE = "gtr-t5-base"
    GTR_T5_LARGE = "gtr-t5-large"
    E5_BASE = "e5-base"
    E5_LARGE = "e5-large"
    DISTILBERT_BASE_UNCASED = "distilbert-base-uncased"
    DISTILUSE_BASE_MULTILINGUAL = "distiluse-base-multilingual-cased-v1"
    CONTRIEVER = "contriever"


class EmbeddingModelInfo(BaseModel):
    """
    A simple class to hold the information for embeddings models
    """

    path: str
    dimensions: int


class EmbeddingModel(BaseModel):
    """
    A class to match the name of an embeddings model with the
    details required to download and use it.
    """

    name: EmbeddingModelName
    info: EmbeddingModelInfo


EMBEDDING_MODELS = {
    # ------ Bidirectional Gated Encoder  ------- >
    # ------ Small { BGESMALL } ------- >
    EmbeddingModelName.BGESMALL: EmbeddingModelInfo(
        path="BAAI/bge-small-en-v1.5", dimensions=384
    ),
    # ------ SBERT (Sentence-BERT) ------- >
    # ------ MiniLM { MINILM } ------- >
    EmbeddingModelName.MINILM: EmbeddingModelInfo(
        path="sentence-transformers/all-MiniLM-L6-v2", dimensions=384
    ),
    # ------ Generalizable T5 Retrieval ------- >
    # ------ Base { GTR_T5_BASE } ------- >
    EmbeddingModelName.GTR_T5_BASE: EmbeddingModelInfo(
        path="google/gtr-t5-base", dimensions=768
    ),
    # ------ Generalizable T5 Retrieval ------- >
    # ------ Large { GTR_T5_LARGE } ------- >
    EmbeddingModelName.GTR_T5_LARGE: EmbeddingModelInfo(
        path="google/gtr-t5-large", dimensions=1024
    ),
    # ------ Embedding Models for Search Engines ------- >
    # ------ Base { E5_BASE } ------- >
    EmbeddingModelName.E5_BASE: EmbeddingModelInfo(
        path="microsoft/e5-base", dimensions=768
    ),
    # ------ Embedding Models for Search Engines ------- >
    # ------ Large { E5_LARGE } ------- >
    EmbeddingModelName.E5_LARGE: EmbeddingModelInfo(
        path="microsoft/e5-large", dimensions=1024
    ),
    # ------ DistilBERT ------- >
    # ------ Base Uncased { DISTILBERT_BASE_UNCASED } ------- >
    EmbeddingModelName.DISTILBERT_BASE_UNCASED: EmbeddingModelInfo(
        path="distilbert-base-uncased", dimensions=768
    ),
    # ------ distiluse-base-multilingual-cased-v1 ------- >
    # ------ Base Multilingual { DISTILUSE_BASE_MULTILINGUAL } ------- >
    EmbeddingModelName.DISTILUSE_BASE_MULTILINGUAL: EmbeddingModelInfo(
        path="sentence-transformers/distiluse-base-multilingual-cased-v1",
        dimensions=512,
    ),
    # ------ Contriever ------- >
    # ------ Contriever { CONTRIEVER } ------- >
    EmbeddingModelName.CONTRIEVER: EmbeddingModelInfo(
        path="facebook/contriever", dimensions=768
    ),
}


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
        embeddings_path: str,
        force_rebuild: bool,
        embed_vocab: List[str],
        model_name: EmbeddingModelName,
        search_kwargs: dict,
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
        self.embeddings_path = embeddings_path
        self.model = get_embedding_model(model_name)
        self.embed_vocab = embed_vocab
        self.search_kwargs = search_kwargs

        if force_rebuild or not os.path.exists(embeddings_path):
            self._build_embeddings()
        else:
            self._load_embeddings()

    def _build_embeddings(self):
        """
        Build a vector database of embeddings
        """
        # Create the directory if it doesn't exist
        if os.path.dirname(self.embeddings_path):
            os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)

        self.embeddings_store = QdrantDocumentStore(
            path=self.embeddings_path,
            embedding_dim=self.model.info.dimensions,
            recreate_index=True,  # We're building from scratch, so recreate the index
        )

        load_dotenv()

        DB_HOST = os.environ["DB_HOST"]
        DB_USER = environ["DB_USER"]
        DB_PASSWORD = quote_plus(environ["DB_PASSWORD"])
        DB_NAME = environ["DB_NAME"]
        DB_PORT = environ["DB_PORT"]

        connection_string = (
            f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )

        # Fetch concept names from the database
        engine = create_engine(connection_string)
        with Session(engine) as session:
            concepts = (
                session.query(Concept.concept_name, Concept.concept_id)
                .filter(Concept.vocabulary_id.in_(self.embed_vocab))
                .all()
            )

        # Create documents from concept names
        concept_docs = [
            Document(
                content=concept.concept_name, meta={"concept_id": concept.concept_id}
            )
            for concept in concepts
        ]
        concept_embedder = FastembedDocumentEmbedder(
            model=self.model.info.path, parallel=0
        )
        concept_embedder.warm_up()
        concept_embeddings = concept_embedder.run(concept_docs)
        self.embeddings_store.write_documents(concept_embeddings.get("documents"))

    def _load_embeddings(self):
        """
        If available, load a vector database of concept embeddings
        """
        self.embeddings_store = QdrantDocumentStore(
            path=self.embeddings_path,
            embedding_dim=self.model.info.dimensions,
            recreate_index=False,  # We're loading existing embeddings, don't recreate
        )

    def get_embedder(self) -> FastembedTextEmbedder:
        """
        Get an embedder for queries in LLM pipelines

        Returns
        _______
        FastembedTextEmbedder
        """
        query_embedder = FastembedTextEmbedder(model=self.model.info.path, parallel=0)
        query_embedder.warm_up()
        return query_embedder

    def get_retriever(self) -> QdrantEmbeddingRetriever:
        """
        Get a retriever for LLM pipelines

        Returns
        -------
        QdrantEmbeddingRetriever
        """
        print(self.search_kwargs)
        return QdrantEmbeddingRetriever(
            document_store=self.embeddings_store, **self.search_kwargs
        )

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
        retriever = QdrantEmbeddingRetriever(document_store=self.embeddings_store)
        query_embedder = FastembedTextEmbedder(
            model=self.model.info.path, parallel=0, prefix="query:"
        )
        query_embedder.warm_up()
        query_embeddings = [query_embedder.run(name) for name in query]
        result = [
            retriever.run(query_embedding["embedding"], **self.search_kwargs)
            for query_embedding in query_embeddings
        ]
        return [
            [
                {
                    "concept_id": doc.meta["concept_id"],
                    "concept": doc.content,
                    "score": doc.score,
                }
                for doc in res["documents"]
            ]
            for res in result
        ]
