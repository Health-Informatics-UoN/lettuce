import asyncio
from collections.abc import AsyncGenerator
import json
from enum import Enum
from typing import Optional, List, Dict, Any


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

import assistant
from omop import OMOP_match
from options.base_options import BaseOptions
from utils.logging_utils import Logger
from components.embeddings import Embeddings
from components.embeddings import EmbeddingModel

logger = Logger().make_logger()
app = FastAPI(
    title="OMOP concpet Assistant",
    description="The API to assist in identifying OMOP concepts",
    version="0.1.0",
    contact={
        "name": "Reza Omidvar",
        "email": "reza.omidvar@nottingham.ac.uk",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# the LLMModel class could be more powerful if we remove the option of using the GPTs.
# Then it can be an Enum of dicts, and the dicts can be unpacked into the arguments for the hf download

class LLMModel(str, Enum):
    """
    This enum holds the names of the different models the assistant can use
    """

    GPT_3_5_TURBO = "gpt-3.5-turbo-0125",
    GPT_4 = "gpt-4",
    LLAMA_2_7B = "llama-2-7B-chat",
    LLAMA_3_8B = "llama-3-8B",
    LLAMA_3_70B = "llama-3-70B",
    GEMMA_7B = "gemma-7b"


class PipelineOptions(BaseModel):
    """
    This class holds the options available to the Llettuce pipeline

    These are all the options in the BaseOptions parser. The defaults provided here match the default options in BaseOptions. Using a pydantic model means FastAPI can take these as input in the API request

    Attributes
    ----------
    llm_model: LLMModel
        The name of the LLM used in the pipeline. The permitted values are the possibilities in the LLMModel enum
    temperature: float
        Temperature supplied to the LLM that tunes the variability of responses
    concept_ancestor: bool
        If true, the concept_ancestor table of the OMOP vocabularies is queried for the results of an OMOP search. Defaults to false
    concept_relationship: bool
        If true, the concept_relationship table of the OMOP vocabularies is queried for the results of an OMOP search. Defaults to false
    concept_synonym: bool
        If true, the concept_synonym table of the OMOP vocabularies is queried when OMOP concepts are fetched. Defaults to false
    search_threshold: int
        The threshold on fuzzy string matching for returned results
    max_separation_descendant: int
        The maximum separation to search for concept descendants
    max_separation_ancestor: int
        The maximum separation to search for concept ancestors
    """

    llm_model: LLMModel = LLMModel.LLAMA_3_8B
    temperature: float = 0
    vocabulary_id: str = "RxNorm"  # TODO: make multiples possible
    concept_ancestor: bool = False
    concept_relationship: bool = False
    concept_synonym: bool = False
    search_threshold: int = 80
    max_separation_descendants: int = 1
    max_separation_ancestor: int = 1
    embeddings_path: str = "concept_embeddings.qdrant"
    force_rebuild: bool = False
    embed_vocab: list[str] = ["RxNorm", "RxNorm Extension"]
    embedding_model: EmbeddingModel = EmbeddingModel.BGESMALL
    embedding_search_kwargs: dict = {}


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
    pipeline_options: Optional[PipelineOptions] = Field(default_factory=PipelineOptions)


def parse_pipeline_args(base_options: BaseOptions, options: PipelineOptions) -> None:
    """
    Use the values of a PipelineOptions object to override defaults

    Parameters
    ----------
    base_options: BaseOptions
        The base options from the command-line application
    options: PipelineOptions
        Overrides from an API request

    Returns
    -------
    None
    """
    base_options._parser.set_defaults(
        llm_model=options.llm_model.value,
        temperature=options.temperature,
        vocabulary_id=options.vocabulary_id,
        concept_ancestor="y" if options.concept_ancestor else "n",
        concept_relationship="y" if options.concept_relationship else "n",
        concept_synonym="y" if options.concept_synonym else "n",
        search_threshold=options.search_threshold,
        max_separation_descendants=options.max_separation_descendants,
        max_separation_ancestor=options.max_separation_ancestor,
    )


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


@app.post("/run")
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


@app.post("/run_db")
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
    
@app.post("/run_vector_search")
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
            model=request.pipeline_options.embedding_model,
            search_kwargs=request.pipeline_options.embedding_search_kwargs,
            )
    return {'event': 'vector_search_output', 'content': embeddings.search(search_terms)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
