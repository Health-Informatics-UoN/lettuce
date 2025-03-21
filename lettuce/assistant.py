from logging import Logger
import time

from dotenv import load_dotenv

from components.pipeline import LLMPipeline
from options.pipeline_options import LLMModel


def run(
    llm_model: LLMModel,
    temperature: float,
    informal_names: list[str],
    logger: Logger,
) -> list[dict]:
    """
    Run the LLM assistant to suggest a formal drug name for an informal medicine name

    Parameters
    ----------
    llm_model: LLMModel
        Choice of model to run
    temperature: float
        Temperature to use for generation
    informal_names: list[str]
        The informal names of the medications
    logger: Logger
        The logger to use

    Returns
    -------
    dict or None
        A dictionary containing the assistant's output

        - 'reply': str, the formal names suggested by the assistant
        - 'meta': dict, metadata from an LLM Generator

        Returns None if no informal_name is provided

    """
    run_start = time.time()
    load_dotenv()

    pipeline = LLMPipeline(
        llm_model=llm_model, temperature=temperature, logger=logger
    ).get_simple_assistant()
    start = time.time()
    pipeline.warm_up()
    logger.info(f"Pipeline warmup in {time.time()-start} seconds")

    results = []

    for informal_name in informal_names:
        start = time.time()
        res = pipeline.run({"prompt": {"informal_name": informal_name}})
        replies = res["llm"]["replies"][0].strip()
        meta = res["llm"]["meta"]
        logger.info(f"Pipeline run in {time.time()-start} seconds")

        logger.info(f"Reply: {replies}")
        logger.info(f"Meta: {meta}")

        output = {"reply": replies, "informal_name": informal_name, "meta": meta}
        logger.info(f"LLM Output: {output}")
        results.append(output)

    logger.info(f"Complete run in {time.time()-run_start} seconds")
    return results


if __name__ == "__main__":
    from options.base_options import BaseOptions
    from utils.logging_utils import logger

    opt = BaseOptions().parse()
    informal_names = opt.informal_names
    run(
        llm_model=opt.LLMModel,
        temperature=opt.temperature,
        informal_names=informal_names,
        logger=logger,
    )
