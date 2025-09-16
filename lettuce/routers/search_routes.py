from typing import Annotated, List
from components.embeddings import EmbeddingModelName, Embeddings
from fastapi import APIRouter, Query

from api_models.responses import ConceptSuggestionResponse, Suggestion, SuggestionsMetaData
from components.pipeline import LLMPipeline
from omop.db_manager import get_session
from omop.omop_queries import count_concepts, query_ids_matching_name, ts_rank_query
from utils.logging_utils import logger
from options.base_options import BaseOptions

settings = BaseOptions()

router = APIRouter()

@router.get("/")
def check_db():
    with get_session() as session:
        query = count_concepts()
        result = session.execute(query).first()
    return f"There are {result[0]} concepts"

@router.get("/text-search/{search_term}")
async def text_search(
        search_term: str,
        vocabulary: Annotated[List[str] | None, Query()]=None,
        domain: Annotated[List[str] | None, Query()]=None,
        standard_concept: bool=True,
        valid_concept: bool=True,
        top_k: Annotated[int, Query(title="The number of responses to fetch", ge=1)]=5,
        ) -> ConceptSuggestionResponse:
    if top_k:
        top_k = top_k
    query = ts_rank_query(
            search_term=search_term,
            vocabulary_id=vocabulary,
            domain_id=domain,
            standard_concept=standard_concept,
            valid_concept=valid_concept,
            top_k=top_k,
            )
    with get_session() as session:
        results = session.execute(query).fetchall()

    metadata = SuggestionsMetaData(pipeline="Full-text search")
    response = ConceptSuggestionResponse(
            items=[
                Suggestion(
                    conceptName=r.concept_name,
                    conceptId=r.concept_id,
                    conceptCode=r.concept_code,
                    domain=r.domain_id,
                    vocabulary=r.vocabulary_id,
                    conceptClass=r.concept_class_id,
                    standard_concept=r.standard_concept,
                    invalid_reason=r.invalid_reason,
                    ranks={"text_search": i+1},
                    scores={"text_search": r.ts_rank},
                    ) for i, r in enumerate(results)
                ],
            metadata=metadata
            )
    return response

@router.get("/vector-search/{search_term}")
async def vector_search(
        search_term: str,
        vocabulary: Annotated[List[str] | None, Query()]=None,
        domain: Annotated[List[str] | None, Query()]=None,
        standard_concept: bool=True,
        valid_concept: bool=False,
        top_k: Annotated[int, Query(title="The number of responses to fetch", ge=1)]=5,
        ) -> ConceptSuggestionResponse:
    embedding_handler = Embeddings(
            model_name=EmbeddingModelName.BGESMALL,
            embed_vocab=vocabulary,
            domain_id=domain,
            standard_concept=standard_concept,
            valid_concept=valid_concept,
            top_k=top_k
            )
    embedder = embedding_handler.get_embedder()
    embedding = embedder.run(search_term)
    retriever = embedding_handler.get_retriever()
    result = retriever.run(embedding["embedding"], describe_concept=True)
    return ConceptSuggestionResponse(
            items=[
                Suggestion(
                    conceptName=r.Concept.concept_name,
                    conceptId=r.Concept.concept_id,
                    conceptCode=r.Concept.concept_code,
                    domain=r.Concept.domain_id,
                    vocabulary=r.Concept.vocabulary_id,
                    conceptClass=r.Concept.concept_class_id,
                    standard_concept=r.Concept.standard_concept,
                    invalid_reason=r.Concept.invalid_reason,
                    ranks={"vector-search": i+1},
                    scores={"vector-search": r.score}
                    )
                for i,r in enumerate(result)
                ],
            metadata=SuggestionsMetaData(pipeline="vector search")
            )
    

@router.get("/ai-search/{search_term}")
async def ai_search(
        search_term: str,
        vocabulary: Annotated[List[str] | None, Query()]=None,
        domain: Annotated[List[str] | None, Query()]=None,
        standard_concept: bool=True,
        valid_concept: bool=False,
        top_k: Annotated[int, Query(title="The number of responses to fetch", ge=1)]=5,
        ) -> ConceptSuggestionResponse:
    assistant = LLMPipeline(
            llm_model=settings.llm_model,
            temperature=0,
            logger=logger,
            embed_vocab=vocabulary,
            standard_concept=standard_concept,
            ).get_rag_assistant()
    answer = assistant.run(
            {
                "prompt": {"informal_name": search_term, "domain": domain},
                "query_embedder": {"text": search_term}
                },
            include_outputs_from="prompt"
            )
    reply = answer["llm"]["replies"][0].strip()
    meta = answer["llm"]["meta"]
    logger.info(f"Reply: {reply}")
    logger.info(f"Meta: {meta}")
    query = query_ids_matching_name(
            query_concept=reply,
            vocabulary_ids=vocabulary,
            full_concept=True
            )
    suggestion_info = {
                "LLM": settings.llm_model.value,
                "LLM reply": reply,
                }

    if settings.debug_prompt:
        suggestion_info["prompt"] = answer["prompt"]

    metadata = SuggestionsMetaData(
            pipeline="LLM RAG pipeline",
            info=suggestion_info
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
        response = ConceptSuggestionResponse(
            items=[
                Suggestion(
                    conceptName=r.concept_name,
                    conceptId=r.concept_id,
                    conceptCode=r.concept_code,
                    domain=r.domain_id,
                    vocabulary=r.vocabulary_id,
                    conceptClass=r.concept_class_id,
                    standard_concept=r.standard_concept,
                    invalid_reason=r.invalid_reason,
                    ranks={"text_search": i+1},
                    scores={"text_search": r.ts_rank},
                    ) for i, r in enumerate(results)
                ],
            metadata=metadata
            )
    else:
        response = ConceptSuggestionResponse(
            items=[
                Suggestion(
                    conceptName=r.concept_name,
                    conceptId=r.concept_id,
                    conceptCode=r.concept_code,
                    domain=r.domain_id,
                    vocabulary=r.vocabulary_id,
                    conceptClass=r.concept_class_id,
                    standard_concept=r.standard_concept,
                    invalid_reason=r.invalid_reason,
                    ranks={},
                    scores={}
                    ) for r in results
                ],
            metadata=metadata
            )
    
    return response
