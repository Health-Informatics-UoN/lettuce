import time
from typing import List, Optional
from typing_extensions import Annotated

from components.embeddings import Embeddings, EmbeddingModelName
from components.pipeline import LLMPipeline
from components.result import LettuceResult
from options.base_options import BaseOptions
from options.pipeline_options import LLMModel
from omop.omop_match import OMOPMatcher
from utils.logging_utils import logger

import typer

app = typer.Typer()

@app.command()
def search(
        informal_names: Annotated[List[str], typer.Argument(help="Source term to search for")],
        vector_search: Annotated[bool, typer.Option(help="Whether to enable vector search in your pipeline")] = True,
        use_llm: Annotated[bool, typer.Option(help="Whether to enable the LLM step in your pipeline")] = True,
        vocabulary_id: Annotated[Optional[List[str]], typer.Option(help="Which vocabularies to return OMOP concepts from")] = None,
        embed_vocab: Annotated[Optional[List[str]], typer.Option(help="Which vocabularies to use for semantic search")] = None,
        standard_concept: Annotated[bool, typer.Option(help="Whether to search through only standard concepts")] = True,
        search_threshold: Annotated[int, typer.Option(help="What fuzzy matching threshold to limit responses to")] = 80,
        verbose_llm: Annotated[bool, typer.Option(help="Whether the LLM should report on its state while initializing")] = False
        ):
    """
    Start a Lettuce search pipeline
    """
    settings = BaseOptions()

    results = [LettuceResult(name) for name in informal_names]

    if vector_search and use_llm:
        start = time.time()
        pl = LLMPipeline(
            llm_model=settings.llm_model,
            temperature=settings.temperature,
            logger=logger,
            embed_vocab=embed_vocab,
            standard_concept=standard_concept,
            embedding_model=settings.embedding_model,
            top_k=settings.embedding_top_k,
            verbose_llm=verbose_llm,
        ).get_rag_assistant()
        pl.warm_up()
        logger.info(f"Pipeline warmup in {time.time() - start} seconds")

        run_start = time.time()

        for query in results:
            rag = pl.run(
                {
                    "query_embedder": {"text": query.search_term},
                    "prompt": {"informal_name": query.search_term},
                },
                include_outputs_from={"retriever", "llm"},
            )
            query.add_vector_search_results(
                [
                    {"content": doc.content, "score": doc.score}
                    for doc in rag["retriever"]["documents"]
                ]
            )
            if "llm" in rag.keys():
                query.add_llm_answer(rag["llm"]["replies"][0].strip())
        logger.info(f"Total RAG inference time: {time.time()-run_start}")
    elif vector_search:
        embeddings = Embeddings(
            model_name=settings.embedding_model,
            embed_vocab=embed_vocab,
            standard_concept=standard_concept,
            top_k=settings.embedding_top_k,
       )
        embed_results = embeddings.search(informal_names)
        for query, result in zip(results, embed_results):
            query.add_vector_search_results(result)
    elif use_llm:
        run_start = time.time()
        pipeline = LLMPipeline(
            llm_model=settings.llm_model,
            temperature=settings.temperature,
            logger=logger,
        ).get_simple_assistant()
        pipeline.warm_up()

        for query in results:
            res = pipeline.run({"prompt": {"informal_name": query.search_term}})
            query.add_llm_answer(res["llm"]["replies"][0].strip())

    db_queries = [query.get_query() for query in results]

    db_results = OMOPMatcher(
        logger, 
        vocabulary_id= vocabulary_id,
        search_threshold=search_threshold
    ).run(search_terms=db_queries)

    for query, result in zip(results, db_results):
        query.add_matches(result, search_threshold)

    print([result.to_dict() for result in results])


if __name__ == "__main__":
    typer.run(main)
