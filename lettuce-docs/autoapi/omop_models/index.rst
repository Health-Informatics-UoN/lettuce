:py:mod:`omop_models`
=====================

.. py:module:: omop_models


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   omop_models.Concept
   omop_models.ConceptSynonym



Functions
~~~~~~~~~

.. autoapisummary::

   omop_models.build_query



Attributes
~~~~~~~~~~

.. autoapisummary::

   omop_models.DB_SCHEMA
   omop_models.Base


.. py:data:: DB_SCHEMA

   

.. py:data:: Base

   

.. py:class:: Concept


   Bases: :py:obj:`Base`

   This class represents an ORM mapping to the OMOP concept table

   .. py:attribute:: __tablename__
      :value: 'concept'

      

   .. py:attribute:: __table_args__

      

   .. py:attribute:: concept_id

      

   .. py:attribute:: concept_name

      

   .. py:attribute:: vocabulary_id

      

   .. py:attribute:: concept_code

      

   .. py:attribute:: standard_concept

      

   .. py:method:: __repr__() -> str



.. py:class:: ConceptSynonym


   Bases: :py:obj:`Base`

   This class represents an ORM mapping to the OMOP concept_synonym table

   .. py:attribute:: __tablename__
      :value: 'concept_synonym'

      

   .. py:attribute:: __table_args__

      

   .. py:attribute:: concept_id

      

   .. py:attribute:: concept_synonym_name

      

   .. py:attribute:: language_concept_id

      

   .. py:method:: __repr__() -> str



.. py:function:: build_query(search_term: str, vocabulary_id: list[str] | None, concept_synonym: str) -> sqlalchemy.sql.Select

   Builds an OMOP query to search for concepts

   Uses the ORM models for the concept and concept_synonym tables to build a query

   Parameters
   ----------
   search_term: str
       The term to use when searching the relevant tables for concepts
   vocabulary_id: list[str]
       A list of vocabulary_ids in the concepts table. The returned concepts will have one of these vocabulary_ids
   concept_synonym: str
       If 'y', then the query is expanded to find matches using the concept_synonym table

   Returns
   -------
   Select
       An SQLAlchemy Select for the desired query


