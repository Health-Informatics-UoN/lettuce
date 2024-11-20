:py:mod:`ui_utilities`
======================

.. py:module:: ui_utilities


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   ui_utilities.display_concept_info
   ui_utilities.stream_message
   ui_utilities.capitalize_words
   ui_utilities.make_api_call



.. py:function:: display_concept_info(concept: dict) -> None

   Display the concept information.
   An OMOP concept is formatted as HTML to be streamed to the user.

   Parameters
   ----------
   concept: dict
       The concept information


.. py:function:: stream_message(message: str) -> None

   Stream a message to the user, rendering HTML with a typewriter effect

   The message is displayed character by character. Streamlit's markdown functionality renders the message, allowing HTML formatting.

   Parameters
   ----------
   message: str
       The message to stream


.. py:function:: capitalize_words(s: str) -> str

   Capitalize each word in a string

   Parameters
   ----------
   s: str
       The string to capitalize

   Returns
   -------
   str
       The capitalized string


.. py:function:: make_api_call(names: list[str], skip_llm: bool, vocab_id: str | None) -> sseclient.SSEClient

   Make a call to the Lettuce API to retrieve OMOP concepts.

   Parameters
   ----------
   names: list[str]
       The informal names to send to the API

   Returns
   -------
   sseclient.SSEClient
       The stream of events from the API


