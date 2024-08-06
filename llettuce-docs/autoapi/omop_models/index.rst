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


