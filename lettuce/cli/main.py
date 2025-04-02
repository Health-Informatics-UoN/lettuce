import time

from components.embeddings import Embeddings, EmbeddingModelName
from components.pipeline import LLMPipeline
from components.result import LettuceResult
from options.base_options import BaseOptions
from options.pipeline_options import LLMModel
from omop.OMOP_match import run 
from utils.logging_utils import logger


def main():
    opt = BaseOptions()
    opt.initialize()
    args = opt.parse()

    results = [LettuceResult(name) for name in args.informal_names]

    if args.vector_search and args.use_llm:
        start = time.time()
        pl = LLMPipeline(
            llm_model=LLMModel[args.llm_model],
            temperature=args.temperature,
            logger=logger,
            embed_vocab=args.embed_vocab,
            standard_concept=args.standard_concept,
            embedding_model=args.embedding_model,
            top_k=args.embedding_top_k,
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
    elif args.vector_search:
        embeddings = Embeddings(
            model_name=EmbeddingModelName.BGESMALL,
            embed_vocab=args.embed_vocab,
            standard_concept=args.standard_concept,
            top_k=args.embedding_top_k,
       )
        embed_results = embeddings.search(args.informal_names)
        for query, result in zip(results, embed_results):
            query.add_vector_search_results(result)
    elif args.use_llm:
        run_start = time.time()
        pipeline = LLMPipeline(
            llm_model=LLMModel[args.llm_model],
            temperature=args.temperature,
            logger=logger,
        ).get_simple_assistant()
        pipeline.warm_up()

        for query in results:
            res = pipeline.run({"prompt": {"informal_name": query.search_term}})
            query.add_llm_answer(res["llm"]["replies"][0].strip())

    db_queries = [query.get_query() for query in results]

    db_results = run(
        search_term=db_queries,
        logger=logger,
        vocabulary_id=args.vocabulary_id,
        search_threshold=args.search_threshold,
    )

    for query, result in zip(results, db_results):
        query.add_matches(result, args.search_threshold)

    print([result.to_dict() for result in results])
