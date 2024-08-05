import argparse
import time

from dotenv import load_dotenv

from components.pipeline import llm_pipeline
from utils.logging_utils import Logger
from utils.utils import *


def run(
    opt: argparse.Namespace = None,
    informal_names: list[str] = None,
    logger: Logger | None = None,
) -> list[dict] | None:
    """
    Run the LLM assistant to suggest a formal drug name for an informal medicine name

    Parameters
    ----------
    opt: argparse.Namespace
        The options for the assistant
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
    if logger is None:
        logger = Logger().make_logger()

    if not informal_names:
        return

    pipeline = llm_pipeline(opt=opt, logger=logger).get_simple_assistant()
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

    opt = BaseOptions().parse()
    informal_names = opt.informal_names
    run(opt=opt, informal_names=informal_names)
