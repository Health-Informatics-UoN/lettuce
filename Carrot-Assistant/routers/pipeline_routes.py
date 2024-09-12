from fastapi import APIRouter, Request
import asyncio
from collections.abc import AsyncGenerator
import json
from typing import List, Dict, Any

from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

import assistant
from omop import OMOP_match
from omop.OMOP_match import OMOPMatcher
from options.base_options import BaseOptions
from components.embeddings import Embeddings
from options.pipeline_options import PipelineOptions, parse_pipeline_args
from utils.logging_utils import Logger

router = APIRouter()

logger = Logger().make_logger()


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


async def generate_events(
    request: PipelineRequest, use_llm: bool, end_session: bool
) -> AsyncGenerator[str]:
    """
    Generate LLM output and OMOP results for a list of informal names.

    parameters
    ----------
    request: PipelineRequest
        The request containing the list of informal names.

    use_llm: bool
        A flag to determine whether to use LLM to find the formal name.

    end_session: bool
        A flag to determine whether to end the session.

    Yields
    ------
    str
        JSON encoded strings of the event results.
    """

    informal_names = request.names
    opt = BaseOptions()
    opt.initialize()
    parse_pipeline_args(opt, request.pipeline_options)
    opt = opt.parse()

    print("Received informal names:", informal_names)
    print(f"use_llm flag is set to: {use_llm}")
    print(f"end_session flag is set to: {end_session}")
    
    
    # If the user chooses to end the session, close the database connection
    if end_session:
        print("Final API call. Closing the database connection....")
        output = {"event": "session_ended", "message": "Session has ended."}
        yield json.dumps(output)
        OMOPMatcher.get_instance().close() 
        return 

    no_match_names = []

    try:
        if informal_names:

            # Query OMOP for the informal names
            if not use_llm:
                for informal_name in informal_names:
                    print(f"Querying OMOP for informal name: {informal_name}")
                    omop_output = OMOP_match.run(
                        opt=opt, search_term=informal_name, logger=logger
                    )

                    # If a match is found, yield the OMOP output
                    if omop_output and any(
                        concept["CONCEPT"] for concept in omop_output
                    ):
                        print(f"OMOP match found for {informal_name}: {omop_output}")
                        output = {"event": "omop_output", "data": omop_output}
                        yield json.dumps(output)

                    # If no match is found, yield a message and add the name to the no_match_names list
                    else:
                        print(f"No satisfactory OMOP results found for {informal_name}")
                        output = {
                            "event": "omop_output",
                            "data": omop_output,
                            "message": f"No match found in OMOP database for {informal_name}.",
                        }
                        yield json.dumps(output)
                        no_match_names.append(informal_name)
                        print(f"\nno_match_names: {no_match_names}\n")
            else:
                no_match_names = informal_names

            # Use LLM to find the formal name and query OMOP for the LLM output
            if no_match_names and use_llm:
                llm_outputs = assistant.run(
                    opt=opt, informal_names=no_match_names, logger=logger
                )

                for llm_output in llm_outputs:
                    print(
                        "LLM output for",
                        llm_output["informal_name"],
                        ":",
                        llm_output["reply"],
                    )

                    output = {"event": "llm_output", "data": llm_output}
                    yield json.dumps(output)

    finally:

        # Ensure database connection is closed at the end of processing
        if not no_match_names:
            print(
                "no matches found. Closing the database connection..."
            )
            OMOPMatcher.get_instance().close()
        
        else:
            print("\nDatabase connection remains open.")


@router.post("/")
async def run_pipeline(request: Request) -> EventSourceResponse:
    """
    This function runs the pipeline for a list of informal names.

    Parameters
    ----------
    request: Request
        The request containing the list of informal names.

    Workflow
    --------
    The function generates events for each informal name in the list.

    use_llm: bool
        A flag to determine whether to use LLM to find the formal name.

    Returns
    -------
    EventSourceResponse
        The response containing the results of the pipeline.
    """
    body = await request.json()
    pipeline_request = PipelineRequest(**body)

    use_llm = body.get("use_llm", False)
    end_session = body.get("end_session", False)

    print(
        f"Running pipeline with use_llm: {use_llm} and end_session: {end_session}"
    )
    return EventSourceResponse(
        generate_events(pipeline_request, use_llm, end_session)
    )


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
    opt = BaseOptions()
    opt.initialize()
    parse_pipeline_args(opt, request.pipeline_options)
    opt = opt.parse()

    omop_outputs = []
    for search_term in search_terms:
        omop_output = OMOP_match.run(opt=opt, search_term=search_term, logger=logger)
        omop_outputs.append({"event": "omop_output", "content": omop_output})

    return omop_outputs


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
        embeddings_path=request.pipeline_options.embeddings_path,
        force_rebuild=request.pipeline_options.force_rebuild,
        embed_vocab=request.pipeline_options.embed_vocab,
        model_name=request.pipeline_options.embedding_model,
        search_kwargs=request.pipeline_options.embedding_search_kwargs,
    )
    return {"event": "vector_search_output", "content": embeddings.search(search_terms)}
