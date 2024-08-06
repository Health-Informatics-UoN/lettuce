import asyncio
from collections.abc import AsyncGenerator
import json
from enum import Enum
from typing import Optional

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
    title="Medication Assistant",
    description="The API to assist in identifying medications",
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

# the LLMModel class could be more powerful if we remove the option of using the GPTs. Then it can be an Enum of dicts, and the dicts can be unpacked into the arguments for the hf download
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
    vocabulary_id: str = "RxNorm" # TODO: make multiples possible
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
    name: str
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
        concept_ancestor='y' if options.concept_ancestor else 'n',
        concept_relationship='y' if options.concept_relationship else 'n',
        concept_synonym='y' if options.concept_synonym else 'n',
        search_threshold=options.search_threshold,
        max_separation_descendants=options.max_separation_descendants,
        max_separation_ancestor=options.max_separation_ancestor
    )


async def generate_events(request: PipelineRequest) -> AsyncGenerator[str]:
    """
    Generate LLM output and OMOP results for an informal medication name

    The first event is the reply from the LLM
    The second event fetches relevant concepts from the OMOP database using the LLM output

    The function yields results as they become available, allowing for real-time streaming.

    Parameters
    ----------
    request: InformalNameRequest
        The request containing the informal name of the medication

    Yields
    ------
    str
        JSON encoded strings of the event results. Two types are yielded:
        1. "llm_output": The result from the language model processing.
        2. "omop_output": The result from the OMOP database matching.

    """
    informal_name = request.name
    opt = BaseOptions()
    opt.initialize()
    parse_pipeline_args(opt, request.pipeline_options)
    opt=opt.parse()

    llm_output = assistant.run(opt=opt, informal_name=informal_name, logger=logger)
    output = {"event": "llm_output", "data": llm_output}
    yield json.dumps(output)

    # Simulate some delay before sending the next part
    await asyncio.sleep(2)

    omop_output = OMOP_match.run(
        opt=opt, search_term=llm_output["reply"], logger=logger
    )
    output = {"event": "omop_output", "data": omop_output}
    yield json.dumps(output)


@app.post("/run")
async def run_pipeline(request: PipelineRequest) -> EventSourceResponse:
    """
    Call generate_events to run the pipeline

    Parameters
    ----------
    request: InformalNameRequest
        The request containing the informal name of the medication

    Returns
    -------
    EventSourceResponse
        The response containing the events
    """
    return EventSourceResponse(generate_events(request))

@app.post("/run_db")
async def run_db(request: PipelineRequest) -> dict:
    """
    Fetch OMOP concepts for a medication name

    Default options can be overridden by the pipeline_options in the request

    Parameters
    ----------
    request: PipelineRequest
        An API request containing a medication name

    Returns
    -------
    dict
        Details of OMOP concept(s) fetched from a database query
    """
    search_term = request.name
    opt = BaseOptions()
    opt.initialize()
    parse_pipeline_args(opt, request.pipeline_options)
    opt = opt.parse()
    return {'event': 'omop_output', 'content': OMOP_match.run(opt=opt, search_term=search_term, logger=logger)}
    
@app.post("/run_vector_search")
async def run_vector_search(request: PipelineRequest):
    search_term = request.name
    embeddings = Embeddings(
            embeddings_path=request.pipeline_options.embeddings_path,
            force_rebuild=request.pipeline_options.force_rebuild,
            embed_vocab=request.pipeline_options.embed_vocab,
            model=request.pipeline_options.embedding_model,
            search_kwargs=request.pipeline_options.embedding_search_kwargs,
            )
    return {'event': 'vector_search_output', 'content': embeddings.search(search_term)}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

