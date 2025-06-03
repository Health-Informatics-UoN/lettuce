from typing import Annotated, List
from fastapi import APIRouter, Query

from api_models.responses import ConceptSuggestionResponse, Suggestion, SuggestionsMetaData
from omop.db_manager import get_session
from omop.omop_queries import count_concepts, ts_rank_query

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
