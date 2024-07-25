pipeline
========

.. py:module:: pipeline


Classes
-------

.. autoapisummary::

   pipeline.llm_pipeline


Module Contents
---------------

.. py:class:: llm_pipeline(opt: argparse.Namespace, logger: logging.Logger | None = None)

   This class is used to generate a pipeline for the model


   .. py:method:: get_simple_assistant() -> haystack.Pipeline

      Get a simple assistant pipeline that connects a prompt with an LLM

      Returns
      -------
      Pipeline
          The pipeline for the assistant



