:py:mod:`embeddings`
====================

.. py:module:: embeddings


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   embeddings.EmbeddingModel
   embeddings.Embeddings




.. py:class:: EmbeddingModel(path: str, dimensions: int)


   Bases: :py:obj:`enum.Enum`

   An Enum for models used to generate concept embeddings

   .. py:attribute:: BGESMALL
      :value: ('BAAI/bge-small-en-v1.5', 384)

      

   .. py:attribute:: MINILM
      :value: ('sentence-transformers/all-MiniLM-L6-v2', 384)

      


.. py:class:: Embeddings(embeddings_path: str, force_rebuild: bool, embed_vocab: List[str], model: EmbeddingModel, search_kwargs: dict)


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



