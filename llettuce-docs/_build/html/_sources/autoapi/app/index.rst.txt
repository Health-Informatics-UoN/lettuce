app
===

.. py:module:: app


Attributes
----------

.. autoapisummary::

   app.logger
   app.app


Classes
-------

.. autoapisummary::

   app.LLMModel
   app.PipelineOptions
   app.PipelineRequest


Functions
---------

.. autoapisummary::

   app.parse_pipeline_args
   app.generate_events
   app.run_pipeline
   app.run_db


Module Contents
---------------

.. py:data:: logger

.. py:data:: app

.. py:class:: LLMModel

   Bases: :py:obj:`str`, :py:obj:`enum.Enum`


   This enum holds the names of the different models the assistant can use


   .. py:attribute:: GPT_3_5_TURBO
      :value: ('gpt-3.5-turbo-0125',)



   .. py:attribute:: GPT_4
      :value: ('gpt-4',)



   .. py:attribute:: LLAMA_2_7B
      :value: ('llama-2-7B-chat',)



   .. py:attribute:: LLAMA_3_8B
      :value: ('llama-3-8B',)



   .. py:attribute:: LLAMA_3_70B
      :value: ('llama-3-70B',)



   .. py:attribute:: GEMMA_7B
      :value: 'gemma-7b'



.. py:class:: PipelineOptions

   Bases: :py:obj:`pydantic.BaseModel`


   This class holds the options available to the Llettuce pipeline

   These are all the options in the BaseOptions parser. The defaults provided here match the default options in BaseOptions. Using a pydantic model means FastAPI can take these as input in the API request

   Attributes
   ----------
   llm_model: LLMModel
       The name of the LLM used in the pipeline. The permitted values are the possibilities in the LLMModel enum
   temperature: float
       Temperature supplied to the LLM that tunes the variability of responses
   concept_ancestor: bool
       If true, the concept_ancestor table of the OMOP vocabularies is queried for the results of an OMOP search. Defaults to false
   concept_relationship: bool
       If true, the concept_relationship table of the OMOP vocabularies is queried for the results of an OMOP search. Defaults to false
   concept_synonym: bool
       If true, the concept_synonym table of the OMOP vocabularies is queried when OMOP concepts are fetched. Defaults to false
   search_threshold: int
       The threshold on fuzzy string matching for returned results
   max_separation_descendant: int
       The maximum separation to search for concept descendants
   max_separation_ancestor: int
       The maximum separation to search for concept ancestors


   .. py:attribute:: llm_model
      :type:  LLMModel


   .. py:attribute:: temperature
      :type:  float
      :value: 0



   .. py:attribute:: vocabulary_id
      :type:  str
      :value: 'RxNorm'



   .. py:attribute:: concept_ancestor
      :type:  bool
      :value: False



   .. py:attribute:: concept_relationship
      :type:  bool
      :value: False



   .. py:attribute:: concept_synonym
      :type:  bool
      :value: False



   .. py:attribute:: search_threshold
      :type:  int
      :value: 80



   .. py:attribute:: max_separation_descendants
      :type:  int
      :value: 1



   .. py:attribute:: max_separation_ancestor
      :type:  int
      :value: 1



.. py:class:: PipelineRequest

   Bases: :py:obj:`pydantic.BaseModel`


   This class takes the format of a request to the API

   Attributes
   ----------
   name: str
       The drug name sent to a pipeline
   pipeline_options: Optional[PipelineOptions]
       Optionally, the default values can be overridden by instantiating a PipelineOptions object. If none is supplied, default arguments are used


   .. py:attribute:: name
      :type:  str


   .. py:attribute:: pipeline_options
      :type:  Optional[PipelineOptions]


.. py:function:: parse_pipeline_args(base_options: options.base_options.BaseOptions, options: PipelineOptions) -> None

   Use the values of a PipelineOptions object to override defaults

   Parameters
   ----------
   base_options: BaseOptions
       The base options from the command-line application
   options: PipelineOptions
       Overrides from an API request

   Returns
   -------
   None


.. py:function:: generate_events(request: PipelineRequest) -> collections.abc.AsyncGenerator[str]
   :async:


   Generate LLM output and OMOP results for an informal medication name

   The first event is the reply from the LLM
   The second event fetches relevant concepts from the OMOP database using the LLM output

   The function yields results as they become available, allowing for real-time streaming.

   Parameters
   ----------
   request: InformalNameRequest
       The request containing the informal name of the medication

   Yields
   ------
   str
       JSON encoded strings of the event results. Two types are yielded:
       1. "llm_output": The result from the language model processing.
       2. "omop_output": The result from the OMOP database matching.



.. py:function:: run_pipeline(request: PipelineRequest) -> sse_starlette.sse.EventSourceResponse
   :async:


   Call generate_events to run the pipeline

   Parameters
   ----------
   request: InformalNameRequest
       The request containing the informal name of the medication

   Returns
   -------
   EventSourceResponse
       The response containing the events


.. py:function:: run_db(request: PipelineRequest) -> dict
   :async:


   Fetch OMOP concepts for a medication name

   Default options can be overridden by the pipeline_options in the request

   Parameters
   ----------
   request: PipelineRequest
       An API request containing a medication name

   Returns
   -------
   dict
       Details of OMOP concept(s) fetched from a database query


