:py:mod:`embeddings`
====================

.. py:module:: embeddings


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   embeddings.EmbeddingModelName
   embeddings.EmbeddingModelInfo
   embeddings.EmbeddingModel
   embeddings.Embeddings



Functions
~~~~~~~~~

.. autoapisummary::

   embeddings.get_embedding_model



Attributes
~~~~~~~~~~

.. autoapisummary::

   embeddings.EMBEDDING_MODELS


.. py:class:: EmbeddingModelName


   Bases: :py:obj:`str`, :py:obj:`enum.Enum`

   str(object='') -> str
   str(bytes_or_buffer[, encoding[, errors]]) -> str

   Create a new string object from the given object. If encoding or
   errors is specified, then the object must expose a data buffer
   that will be decoded using the given encoding and error handler.
   Otherwise, returns the result of object.__str__() (if defined)
   or repr(object).
   encoding defaults to sys.getdefaultencoding().
   errors defaults to 'strict'.

   .. py:attribute:: BGESMALL
      :value: 'BGESMALL'

      

   .. py:attribute:: MINILM
      :value: 'MINILM'

      


.. py:class:: EmbeddingModelInfo


   Bases: :py:obj:`pydantic.BaseModel`

   .. py:attribute:: path
      :type: str

      

   .. py:attribute:: dimensions
      :type: int

      


.. py:class:: EmbeddingModel


   Bases: :py:obj:`pydantic.BaseModel`

   .. py:attribute:: name
      :type: EmbeddingModelName

      

   .. py:attribute:: info
      :type: EmbeddingModelInfo

      


.. py:data:: EMBEDDING_MODELS

   

.. py:function:: get_embedding_model(name: EmbeddingModelName) -> EmbeddingModel


.. py:class:: Embeddings(embeddings_path: str, force_rebuild: bool, embed_vocab: List[str], model_name: EmbeddingModelName, search_kwargs: dict)


   This class allows the building or loading of a vector database of concept names. This database can then be used for vector search.

   Methods
   -------
   search:
       Query the attached embeddings database with provided search terms

   .. py:method:: _build_embeddings()

      Build a vector database of embeddings


   .. py:method:: _load_embeddings()

      If available, load a vector database of concept embeddings


   .. py:method:: search(query: List[str]) -> List[List[Dict[str, Any]]]

      Search the attached vector database with a list of informal medications

      Parameters
      ----------
      query: List[str]
          A list of informal medication names

      Returns
      -------
      List[List[Dict[str, Any]]]
          For each medication in the query, the result of searching the vector database



