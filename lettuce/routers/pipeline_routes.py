from fastapi import APIRouter
from collections.abc import AsyncGenerator
import json
from typing import List, Dict, Any
import time

from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

import assistant
from omop.omop_match import OMOPMatcher
from components.embeddings import Embeddings
from components.pipeline import LLMPipeline
from options.pipeline_options import PipelineOptions
from utils.logging_utils import logger

router = APIRouter()


class PipelineRequest(BaseModel):
    """
    This class takes the format of a request to the API

    Attributes
    ----------
    name: str
        The drug name sent to a pipeline
    pipeline_options: Optional[PipelineOptions]
        Optionally, the default values can be overridden by instantiating a PipelineOptions object. If none is supplied, default arguments are used
    """

    names: List[str]
    pipeline_options: PipelineOptions = Field(default_factory=PipelineOptions)


async def generate_events(request: PipelineRequest) -> AsyncGenerator[str]:
    """
    Generate LLM output and OMOP results for a list of informal names

    Parameters
    ----------
    request: PipelineRequest
        The request containing the list of informal names.

    Workflow
    --------
    For each informal name:
        The first event is to Query the OMOP database for a match
        The second event is to fetches relevant concepts from the OMOP database
        Finally,The function yields results as they become available,
        allowing for real-time streaming.

    Conditions
    ----------
    If the OMOP database returns a match, the LLM is not queried

    If the OMOP database does not return a match,
    the LLM is used to find the formal name and the OMOP database is
    queried for the LLM output.

    Finally, the function yields the results for real-time streaming.


    Yields
    ------
    str
        JSON encoded strings of the event results. Two types are yielded:
        1. "llm_output": The result from the language model processing.
        2. "omop_output": The result from the OMOP database matching.
    """
    informal_names = request.names

    print("Received informal names:", informal_names)

    # Use LLM to find the formal name and query OMOP for the LLM output
    pipeline_opts = request.pipeline_options

    llm_outputs = assistant.run(
        llm_model=pipeline_opts.llm_model,
        temperature=pipeline_opts.temperature,
        informal_names=informal_names,
        logger=logger,
    )
    for llm_output in llm_outputs:

        logger.info(
            f"LLM output for {llm_output['informal_name']}: {llm_output['reply']}"
        )

        logger.info("Querying OMOP for LLM output: %s", llm_output["reply"])

        output = {"event": "llm_output", "data": llm_output}
        yield json.dumps(output)

    omop_output = OMOPMatcher(
        logger, 
        vocabulary_id=pipeline_opts.vocabulary_id,
        standard_concept=pipeline_opts.standard_concept,
        concept_ancestor=pipeline_opts.concept_ancestor,
        concept_relationship=pipeline_opts.concept_relationship,
        concept_synonym=pipeline_opts.concept_synonym,
        search_threshold=pipeline_opts.search_threshold,
        max_separation_descendant=pipeline_opts.max_separation_descendants,
        max_separation_ancestor=pipeline_opts.max_separation_ancestor
    ).run(search_terms=[llm_output["reply"] for llm_output in llm_outputs])

    output = [{"event": "omop_output", "data": result} for result in omop_output]
    yield json.dumps(output)


@router.post("/")
async def run_pipeline(request: PipelineRequest) -> EventSourceResponse:
    """
    Call generate_events to run the pipeline

    Parameters
    ----------
    request: PipelineRequest
        The request containing a list of informal names
    Returns
    -------
    EventSourceResponse
        The response containing the events
    """
    return EventSourceResponse(generate_events(request))


@router.post("/db")
async def run_db(request: PipelineRequest) -> List[Dict[str, Any]]:
    """
    Fetch OMOP concepts for a name

    Default options can be overridden by the pipeline_options in the request

    Parameters
    ----------
    request: PipelineRequest
        An API request containing a list of informal names and the options of a pipeline

    Returns
    -------
    dict
        Details of OMOP concept(s) fetched from a database query
    """
    search_terms = request.names
    pipeline_opts = request.pipeline_options

    omop_output = OMOPMatcher(
        logger, 
        vocabulary_id=pipeline_opts.vocabulary_id,
        standard_concept=pipeline_opts.standard_concept,
        concept_ancestor=pipeline_opts.concept_ancestor,
        concept_relationship=pipeline_opts.concept_relationship,
        concept_synonym=pipeline_opts.concept_synonym,
        search_threshold=pipeline_opts.search_threshold,
        max_separation_descendant=pipeline_opts.max_separation_descendants,
        max_separation_ancestor=pipeline_opts.max_separation_ancestor
    ).run(search_terms=search_terms)
    return [{"event": "omop_output", "content": result} for result in omop_output]


@router.post("/vector_search")
async def run_vector_search(request: PipelineRequest):
    """
    Search a vector database for a name

    Default options can be overridden by pipeline_options
    A warning: if you don't have a vector database set up under the embeddings_path, this method will build one for you. This takes a while, an hour using 2.8 GHz intel I7, 16 Gb RAM.

    Parameters
    ----------
    request: PipelineRequest
        An API request containing a list of informal names and the options of a pipeline

    Returns
    -------
    list
        Details of OMOP concept(s) fetched from a vector database query
    """
    search_terms = request.names
    embeddings = Embeddings(
        embed_vocab=request.pipeline_options.embed_vocab,
        model_name=request.pipeline_options.embedding_model,
        standard_concept=request.pipeline_options.standard_concept,
        top_k=request.pipeline_options.embeddings_top_k,
    )
    return {"event": "vector_search_output", "content": embeddings.search(search_terms)}


@router.post("/vector_llm")
async def vector_llm_pipeline(request: PipelineRequest) -> List:
    """
    Run a RAG pipeline that first checks a vector database, then uses an LLM

    This has a conditional router in it that checks whether there's an exact match for the term.
    If there is an exact match, the vector search results are returned.
    If there is not, the vector search results are used for retrieval augmented generation

    Parameters
    ----------
    request: PipelineRequest

    Returns
    -------
    list
    """
    informal_names = request.names

    pl = LLMPipeline(
        llm_model=request.pipeline_options.llm_model,
        temperature=request.pipeline_options.temperature,
        embed_vocab=request.pipeline_options.embed_vocab,
        embedding_model=request.pipeline_options.embedding_model,
        logger=logger,
        standard_concept=request.pipeline_options.standard_concept,
        top_k=request.pipeline_options.embeddings_top_k,
    ).get_rag_assistant()
    start = time.time()
    pl.warm_up()
    logger.info(f"Pipeline warmup in {time.time()-start} seconds")

    results = []
    run_start = time.time()

    for informal_name in informal_names:
        start = time.time()
        res = pl.run(
            {
                "query_embedder": {"text": informal_name},
                "prompt": {"informal_name": informal_name},
            },
            include_outputs_from={"retriever", "llm"},
        )
        inference_time = time.time() - start

        # This should not be here, sorry
        def build_output(informal_name, result, inf_time) -> dict:
            output = {
                "informal_name": informal_name,
                "inference_time": inf_time,
            }
            if "llm" in result.keys():
                output["llm_output"] = result["llm"]["replies"][0].strip()
            output["vector_search_output"] = [
                {"content": doc.content, "score": doc.score}
                for doc in result["retriever"]["documents"]
            ]
            return output

        results.append(build_output(informal_name, res, inference_time))

    logger.info(f"Complete run in {time.time()-run_start} seconds")
    return results
