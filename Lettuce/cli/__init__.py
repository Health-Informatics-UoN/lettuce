import time

import assistant
from components import embeddings
from components.pipeline import llm_pipeline
from components.result import LettuceResult
from options.base_options import BaseOptions
from utils.logging_utils import logger

if __name__ == "__main__":
    opt = BaseOptions()
    opt.initialize()
    args = opt.parse()

    results = [LettuceResult(name) for name in args.informal_names]

    if args.vector_search & args.use_llm:
        start = time.time()
        pl = llm_pipeline(
            args.llm_model,
            args.temperature,
            logger=logger,
            embeddings_path="concept_embeddings.qdrant",
            embed_vocab=["RxNorm"],
            embedding_model=embeddings.EmbeddingModelName.BGESMALL,
        ).get_rag_assistant()
        pl.warm_up()
        logger.info(f"Pipeline warmup in {time.time() - start} seconds")

        rag_results = []
        run_start = time.time()

        for query in results:
            rag = pl.run(
                {
                    "query_embedder": {"text": query.search_term},
                    "prompt": {"informal_name": query.search_term},
                },
                include_outputs_from={"retriever", "llm"},
            )
            query.add_vector_search_results(rag["vector_search_output"])
            if "llm" in rag.keys():
                query.add_llm_answer(rag["llm"]["replies"][0].strip())
        logger.info(f"Total RAG inference time: {time.time()-run_start}")
    elif args.vector_search:
        embeddings = embeddings.Embeddings(
            embeddings_path="concept_embeddings.qdrant",
            force_rebuild=False,
            embed_vocab=["RxNorm"],
            model_name=embeddings.EmbeddingModelName.BGESMALL,
            search_kwargs={},
        )
        embed_results = embeddings.search(args.informal_names)
        for query, result in zip(results, embed_results):
            query.add_vector_search_results(result)
    elif args.use_llm:
        run_start = time.time()
        pipeline = llm_pipeline(
            llm_model=args.llm_model, temperature=args.temperature, logger=logger
        ).get_simple_assistant()
        pipeline.warm_up()

        for query in results:
            res = pipeline.run({"prompt": {"informal_name": query.search_term}})
            query.add_llm_answer(res["llm"]["replies"][0].strip())

    print(results)
