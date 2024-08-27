from fastapi import APIRouter
import asyncio
from collections.abc import AsyncGenerator
import json
from typing import List, Dict, Any

from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

import assistant
from omop import OMOP_match
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
    opt = BaseOptions()
    opt.initialize()
    parse_pipeline_args(opt, request.pipeline_options)
    opt = opt.parse()

    print("Received informal names:", informal_names)
    
    # Query OMOP for each informal name

    for informal_name in informal_names:
        print(f"Querying OMOP for informal name: {informal_name}")
        omop_output = OMOP_match.run(opt=opt, search_term=informal_name, logger=logger)

        if omop_output and any(concept["CONCEPT"] for concept in omop_output):
            print(f"OMOP match found for {informal_name}: {omop_output}")
            output = {"event": "omop_output", "data": omop_output}
            yield json.dumps(output)
            continue

        else:
            print("No satisfactory OMOP results found for {informal_name}, using LLM...")

    # Use LLM to find the formal name and query OMOP for the LLM output

    llm_outputs = assistant.run(opt=opt, informal_names=informal_names, logger=logger)
    for llm_output in llm_outputs:


        print("LLM output for", llm_output["informal_name"], ":", llm_output["reply"])
        
        print("Querying OMOP for LLM output:", llm_output["reply"])

        output = {"event": "llm_output", "data": llm_output}
        yield json.dumps(output)

        # Simulate some delay before sending the next part
        await asyncio.sleep(2)

        omop_output = OMOP_match.run(
            opt=opt, search_term=llm_output["reply"], logger=logger
        )

        print("OMOP output for", llm_output["reply"], ":", omop_output)

        output = {"event": "omop_output", "data": omop_output}
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
async def run_db(request: PipelineRequest) -> List[Dict[str,Any]]:
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
    return {'event': 'vector_search_output', 'content': embeddings.search(search_terms)}


