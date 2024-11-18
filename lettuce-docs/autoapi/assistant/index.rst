:py:mod:`assistant`
===================

.. py:module:: assistant


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   assistant.run



Attributes
~~~~~~~~~~

.. autoapisummary::

   assistant.opt


.. py:function:: run(opt: argparse.Namespace = None, informal_names: list[str] = None, logger: utils.logging_utils.Logger | None = None) -> list[dict] | None

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



.. py:data:: opt

   

