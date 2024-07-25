prompt
======

.. py:module:: prompt


Classes
-------

.. autoapisummary::

   prompt.Prompts


Module Contents
---------------

.. py:class:: Prompts(model_name: str, prompt_type: str | None = 'simple')

   This class is used to generate prompts for the models.


   .. py:method:: get_prompt() -> haystack.components.builders.PromptBuilder | None

      Get the prompt based on the prompt_type supplied to the object.

      Returns
      -------
      PromptBuilder
          The prompt for the model

          - If the _prompt_type of the object is "simple", returns a simple prompt for few-shot learning of formal drug names.



   .. py:method:: _simple_prompt() -> haystack.components.builders.PromptBuilder

      Get a simple prompt

      Returns
      -------
      PromptBuilder
          The simple prompt



