from logging import Logger
from typing import List, Literal
from suggestions import ConceptSuggestion
from omop.omop_queries import ts_rank_query, query_ids_matching_name
from omop.db_manager import get_session
from components.embeddings import Embeddings
from options.pipeline_options import EmbeddingModelName
from components.models import connect_to_ollama
from components.pipeline import LLMPipeline


def _text_search(
    search_term: str,
    domain: List[str] | None,
    vocabulary: List[str] | None,
    standard_concept: bool,
    valid_concept: bool,
    top_k: int,
    ) -> List[ConceptSuggestion]:
    """
    Run a lexical search for a search term with specified paramters

    Parameters
    ----------
    search_term: str
        The string to search for
    domain: List[str] | None
        A list of domains to include in results. If None, then all domains are searched
    vocabulary: List[str] | None = None
        A list of vocabularies to include in results. If None, then all vocabularies are searched
    standard_concept: bool = True,
        If true, only standard concepts returned. Otherwise, non-standard concepts will be included.
    valid_concept: bool = True
        If true, only concepts without invalid_reason returned. Otherwise, all concepts will be included.
    top_k: int
        The number of results to return 

    Returns
    -------
    List[ConceptSuggestion]
        The top_k results from the lexical search
    """
    query = ts_rank_query(
        search_term=search_term,
        domain_id=domain,
        vocabulary_id=vocabulary,
        standard_concept=standard_concept,
        valid_concept=valid_concept,
        top_k=top_k,
    )
    with get_session() as session:
        results = session.execute(query).fetchall()
    return [
        ConceptSuggestion(
            concept_id=r.concept_id,
            concept_name=r.concept_name,
            domain_id=r.domain_id,
            vocabulary_id=r.vocabulary_id,
            standard_concept=r.standard_concept,
        )
        for r in results
    ]

def _vector_search(
         search_term: str,
         domain: List[str] | None,
         vocabulary: List[str] | None,
         standard_concept:bool,
         valid_concept: bool,
         embeddings_model_name: str,
         top_k: int,
         ) -> List[ConceptSuggestion]:
    """
    Run a vector search for a search term with specified paramters

    Parameters
    ----------
    search_term: str
        The string to search for
    domain: List[str] | None
        A list of domains to include in results. If None, then all domains are searched
    vocabulary: List[str] | None = None
        A list of vocabularies to include in results. If None, then all vocabularies are searched
    standard_concept: bool = True,
        If true, only standard concepts returned. Otherwise, non-standard concepts will be included.
    valid_concept: bool = True
        If true, only concepts without invalid_reason returned. Otherwise, all concepts will be included.
    embeddings_model_name: str,
        The short name for an embeddings model to match on the EmbeddingModelName enum
    top_k: int
        The number of results to return 

    Returns
    -------
    List[ConceptSuggestion]
        The top_k results from the vector search
   """
    embedding_model = EmbeddingModelName[embeddings_model_name]
    embedding_handler = Embeddings(
        model_name=embedding_model,
        embed_vocab=vocabulary,
        domain_id=domain,
        standard_concept=standard_concept,
        valid_concept=valid_concept,
        top_k=top_k,
    )
    embedder = embedding_handler.get_embedder()
    embedding = embedder.run(search_term)
    retriever = embedding_handler.get_retriever()
    results = retriever.run(embedding["embedding"], describe_concept=True)
    print(results)
    return [
        ConceptSuggestion(
            concept_id=r.Concept.concept_id,
            concept_name=r.Concept.concept_name,
            domain_id=r.Concept.domain_id,
            vocabulary_id=r.Concept.vocabulary_id,
            standard_concept=r.Concept.standard_concept,
            score=r.score,
        )
        for r in results
    ]

def _ai_search(
        search_term: str,
        domain: List[str] | None,
        vocabulary: List[str] | None,
        standard_concept: bool,
        valid_concept: bool,
        embeddings_model_name: str,
        top_k: int,
        llm_name: str,
        llm_url: str,
        logger: Logger,
        ) -> List[ConceptSuggestion]:
    """
    Run an LLM-powered search for a search term with specified paramters

    Parameters
    ----------
    search_term: str
        The string to search for
    domain: List[str] | None
        A list of domains to include in results. If None, then all domains are searched
    vocabulary: List[str] | None = None
        A list of vocabularies to include in results. If None, then all vocabularies are searched
    standard_concept: bool = True,
        If true, only standard concepts returned. Otherwise, non-standard concepts will be included.
    valid_concept: bool = True
        If true, only concepts without invalid_reason returned. Otherwise, all concepts will be included.
    embeddings_model_name: str,
        The short name for an embeddings model to match on the EmbeddingModelName enum
    top_k: int
        The number of results to return 
    llm_name: str
        The short name for an LLM to match either a recognised name on a server or the locally saved models
    llm_url: str
        The URL for an LLM server
    logger: Logger
        log your problems

    Returns
    -------
    List[ConceptSuggestion]
        The top_k results from the vector search
   """
    llm = connect_to_ollama(
            model_name=llm_name,
            url=llm_url,
            temperature=0.7,
            logger=logger,
        )
    embedding_model = EmbeddingModelName[embeddings_model_name]
    assistant = LLMPipeline(
        llm=llm,
        temperature=0,
        logger=logger,
        embed_vocab=vocabulary,
        standard_concept=standard_concept,
        embedding_model=embedding_model,
    ).get_rag_assistant()
    answer = assistant.run(
        {
            "prompt": {"informal_name": search_term, "domain": domain},
            "query_embedder": {"text": search_term},
        },
    )
    reply = answer["llm"]["replies"][0].strip()
    query = query_ids_matching_name(
        query_concept=reply, vocabulary_ids=vocabulary, full_concept=True
    )
    with get_session() as session:
        results = session.execute(query).fetchall()
        if len(results) == 0:
            ts_query = ts_rank_query(
                search_term=reply,
                vocabulary_id=vocabulary,
                domain_id=domain,
                standard_concept=standard_concept,
                valid_concept=valid_concept,
                top_k=top_k,
            )
            with get_session() as session:
                results = session.execute(ts_query).fetchall()
            return [
                ConceptSuggestion(
                    concept_id=r.Concept.concept_id,
                    concept_name=r.Concept.concept_name,
                    domain_id=r.Concept.domain_id,
                    vocabulary_id=r.Concept.vocabulary_id,
                    standard_concept=r.Concept.standard_concept,
                )
                for r in results
            ]
        return [
            ConceptSuggestion(
                concept_id=r[1],
                concept_name=r[0],
                domain_id=r[3],
                vocabulary_id=r[4],
                standard_concept=r[6],
            )
            for r in results
        ]

    

def search(
    search_term: str,
    domain: List[str] | None,
    vocabulary: List[str] | None,
    standard_concept: bool,
    valid_concept: bool,
    top_k: int,
    search_mode: Literal["text-search", "vector-search", "ai-search"],
    embeddings_model_name: str,
    llm_name: str,
    llm_url: str,
    logger: Logger,
    ) -> List[ConceptSuggestion]:
    """
    Run an LLM-powered search for a search term with specified paramters

    Parameters
    ----------
    search_term: str
        The string to search for
    domain: List[str] | None
        A list of domains to include in results. If None, then all domains are searched
    vocabulary: List[str] | None = None
        A list of vocabularies to include in results. If None, then all vocabularies are searched
    standard_concept: bool = True,
        If true, only standard concepts returned. Otherwise, non-standard concepts will be included.
    valid_concept: bool = True
        If true, only concepts without invalid_reason returned. Otherwise, all concepts will be included.
    embeddings_model_name: str,
        The short name for an embeddings model to match on the EmbeddingModelName enum
    llm_name: str
        The short name for an LLM to match either a recognised name on a server or the locally saved models
    llm_url: str
        The URL for an LLM server
    logger: Logger
        log your problems

    Returns
    -------
    List[ConceptSuggestion]
        The top_k results from the vector search
   """
    if search_mode == "text-search":
        return _text_search(
                search_term=search_term,
                domain=domain,
                vocabulary=vocabulary,
                standard_concept=standard_concept,
                valid_concept=valid_concept,
                top_k=top_k,
                )
    elif search_mode == "vector-search":
        return _vector_search(
                search_term=search_term,
                domain=domain,
                vocabulary=vocabulary,
                standard_concept=standard_concept,
                valid_concept=valid_concept,
                embeddings_model_name=embeddings_model_name,
                top_k=top_k
                )
    elif search_mode == "ai-search":
        return _ai_search(
                search_term=search_term,
                domain=domain,
                vocabulary=vocabulary,
                standard_concept=standard_concept,
                valid_concept=valid_concept,
                top_k=top_k,
                embeddings_model_name=embeddings_model_name,
                llm_name=llm_name,
                llm_url=llm_url,
                logger=logger,
                )
