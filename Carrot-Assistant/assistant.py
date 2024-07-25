import argparse
import time

from dotenv import load_dotenv

from components.pipeline import llm_pipeline
from utils.logging_utils import Logger
from utils.utils import *


def run(
    opt: argparse.Namespace = None,
    informal_name: str = None,
    logger: Logger | None = None,
) -> dict | None:
    """
    Run the LLM assistant to suggest a formal drug name for an informal medicine name



    Parameters
    ----------
    opt: argparse.Namespace
        The options for the assistant
    informal_name: str
        The informal name of the medication
    logger: Logger
        The logger to use

    Returns
    -------
    dict or None
        A dictionary containing the assistant's output

        - 'reply': str, the formal name suggested by the assistant
        - 'meta': dict, metadata from an LLM Generator

        Returns None if no informal_name is provided
    
    """
    run_start = time.time()
    load_dotenv() # I don't think there's any need to load the .env here is there? JMW
    if logger is None:
        logger = Logger().make_logger()

    if not informal_name:
        return
    pipeline = llm_pipeline(opt=opt, logger=logger).get_simple_assistant()
    start = time.time()
    pipeline.warm_up()
    logger.info(f"Pipeline warmup in {time.time()-start} seconds")
    start = time.time()

    res = pipeline.run({"prompt": {"informal_name": informal_name}})
    replies = res["llm"]["replies"][0].strip()
    meta = res["llm"]["meta"]
    logger.info(f"Pipeline run in {time.time()-start} seconds")
    start = time.time()

    logger.info(f"Reply: {replies}")
    logger.info(f"Meta: {meta}")

    logger.info(f"Complete run in {time.time()-run_start} seconds")

    output = {"reply": replies, "meta": meta}
    logger.info(f"LLM Output: {output}")
    return output


if __name__ == "__main__":
    from options.base_options import BaseOptions

    opt = BaseOptions().parse()
    informal_name = opt.informal_name
    run(opt=opt, informal_name=informal_name)
