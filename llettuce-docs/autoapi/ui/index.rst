:py:mod:`ui`
============

.. py:module:: ui


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   ui.stream_message
   ui.capitalize_words
   ui.make_api_call
   ui.display_concept_info



Attributes
~~~~~~~~~~

.. autoapisummary::

   ui.informal_name
   ui.skip_llm
   ui.result_stream


.. py:data:: informal_name

   

.. py:data:: skip_llm

   

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


.. py:function:: make_api_call(name: str) -> sseclient.SSEClient

   Make a call to the Llettuce API to retrieve OMOP concepts.

   Parameters
   ----------
   name: str
       The informal name to send to the API

   Returns
   -------
   sseclient.SSEClient
       The stream of events from the API


.. py:function:: display_concept_info(concept: dict) -> None

   Display the concept information.
   An OMOP concept is formatted as HTML to be streamed to the user.

   Parameters
   ----------
   concept: dict
       The concept information


.. py:data:: result_stream
   :type: sseclient.SSEClient

   

