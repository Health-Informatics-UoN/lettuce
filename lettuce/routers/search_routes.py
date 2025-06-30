from typing import Annotated, List
from components.embeddings import EmbeddingModelName, Embeddings
from fastapi import APIRouter, Query

from api_models.responses import ConceptSuggestionResponse, Suggestion, SuggestionsMetaData
from components.pipeline import LLMPipeline
from omop.db_manager import get_session
from omop.omop_queries import count_concepts, query_ids_matching_name, ts_rank_query
from options.pipeline_options import LLMModel
from utils.logging_utils import logger

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
        vocabulary_id: Annotated[List[str] | None, Query()]=None,
        domain_id: Annotated[List[str] | None, Query()]=None,
        standard_concept: bool=False,
        valid_concept: bool=False,
        top_k: Annotated[int, Query(title="The number of responses to fetch", ge=1)]=5,
        ) -> ConceptSuggestionResponse:
    if top_k:
        top_k = top_k
    query = ts_rank_query(
            search_term=search_term,
            vocabulary_id=vocabulary_id,
            domain_id=domain_id,
            standard_concept=standard_concept,
            valid_concept=valid_concept,
            top_k=top_k,
            )
    with get_session() as session:
        results = session.execute(query).fetchall()

    metadata = SuggestionsMetaData(pipeline="Full-text search")
    response = ConceptSuggestionResponse(
            recommendations=[
                Suggestion(
                    concept_name=r.concept_name,
                    concept_id=r.concept_id,
                    domain_id=r.domain_id,
                    vocabulary_id=r.vocabulary_id,
                    concept_class_id=r.concept_class_id,
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
        vocabulary_id: Annotated[List[str] | None, Query()]=None,
        domain_id: Annotated[List[str] | None, Query()]=None,
        standard_concept: bool=False,
        valid_concept: bool=False,
        top_k: Annotated[int, Query(title="The number of responses to fetch", ge=1)]=5,
        ) -> ConceptSuggestionResponse:
    embedding_handler = Embeddings(
            model_name=EmbeddingModelName.BGESMALL,
            embed_vocab=vocabulary_id,
            domain_id=domain_id,
            standard_concept=standard_concept,
            valid_concept=valid_concept,
            top_k=top_k
            )
    embedder = embedding_handler.get_embedder()
    embedding = embedder.run(search_term)
    retriever = embedding_handler.get_retriever()
    result = retriever.run(embedding["embedding"], describe_concept=True)
    return ConceptSuggestionResponse(
            recommendations=[
                Suggestion(
                    concept_name=r.Concept.concept_name,
                    concept_id=r.Concept.concept_id,
                    domain_id=r.Concept.domain_id,
                    vocabulary_id=r.Concept.vocabulary_id,
                    concept_class_id=r.Concept.concept_class_id,
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
        vocabulary_id: Annotated[List[str] | None, Query()]=None,
        domain_id: Annotated[List[str] | None, Query()]=None,
        standard_concept: bool=False,
        valid_concept: bool=False,
        top_k: Annotated[int, Query(title="The number of responses to fetch", ge=1)]=5,
        ) -> ConceptSuggestionResponse:
    assistant = LLMPipeline(
            llm_model=LLMModel.LLAMA_3_1_8B,
            temperature=0,
            logger=logger,
            embed_vocab=vocabulary_id,
            standard_concept=standard_concept,
            ).get_rag_assistant()
    answer = assistant.run({"prompt": {"informal_name": search_term}, "query_embedder": {"text": search_term}})
    reply = answer["llm"]["replies"][0].strip()
    meta = answer["llm"]["meta"]
    logger.info(f"Reply: {reply}")
    logger.info(f"Meta: {meta}")
    query = query_ids_matching_name(
            query_concept=reply,
            vocabulary_ids=vocabulary_id,
            full_concept=True
            )
    metadata = SuggestionsMetaData(
            pipeline="LLM RAG pipeline",
            info={
                "LLM": "Llama 3.1 8b (quantised to 4-bit)",
                "LLM reply": reply,
                }
            )
    with get_session() as session:
        results = session.execute(query).fetchall()
    if len(results) == 0:
        ts_query = ts_rank_query(
                search_term=reply,
                vocabulary_id=vocabulary_id,
                domain_id=domain_id,
                standard_concept=standard_concept,
                valid_concept=valid_concept,
                top_k=top_k,
                )
        with get_session() as session:
            results = session.execute(ts_query).fetchall()
        response = ConceptSuggestionResponse(
            recommendations=[
                Suggestion(
                    concept_name=r.concept_name,
                    concept_id=r.concept_id,
                    domain_id=r.domain_id,
                    vocabulary_id=r.vocabulary_id,
                    concept_class_id=r.concept_class_id,
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
            recommendations=[
                Suggestion(
                    concept_name=r.concept_name,
                    concept_id=r.concept_id,
                    domain_id=r.domain_id,
                    vocabulary_id=r.vocabulary_id,
                    concept_class_id=r.concept_class_id,
                    standard_concept=r.standard_concept,
                    invalid_reason=r.invalid_reason,
                    ranks={},
                    scores={}
                    ) for r in results
                ],
            metadata=metadata
            )
    
    return response
