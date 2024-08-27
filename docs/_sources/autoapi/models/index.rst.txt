:py:mod:`models`
================

.. py:module:: models


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   models.get_model



Attributes
~~~~~~~~~~

.. autoapisummary::

   models.local_models


.. py:data:: local_models

   

.. py:function:: get_model(model_name: str, temperature: float = 0.7, logger: logging.Logger | None = None) -> object

   Get an interface for interacting with an LLM

   Uses Haystack Generators to provide an interface to a model.
   If the model_name is a GPT, then the interface is to a remote OpenAI model. Otherwise, uses a LlamaCppGenerator to start a llama.cpp model and provide an interface.

   Parameters
   ----------
   model_name: str
       The name of the model
   temperature: float
       The temperature for the model
   logger: logging.Logger|None
       The logger for the model

   Returns
   -------
   object
       An interface to generate text using an LLM


