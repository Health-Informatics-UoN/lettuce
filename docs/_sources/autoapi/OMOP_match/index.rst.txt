:py:mod:`OMOP_match`
====================

.. py:module:: OMOP_match


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   OMOP_match.OMOPMatcher



Functions
~~~~~~~~~

.. autoapisummary::

   OMOP_match.run



.. py:class:: OMOPMatcher(logger: Optional[utils.logging_utils.Logger] = None)


   This class retrieves matches from an OMOP database and returns the best

   .. py:method:: close()

      Close the engine connection.


   .. py:method:: calculate_best_matches(search_terms: list[str], vocabulary_id: list | None = None, concept_ancestor: str = 'y', concept_relationship: str = 'y', concept_synonym: str = 'y', search_threshold: int = 0, max_separation_descendant: int = 1, max_separation_ancestor: int = 1) -> list

      Calculate best OMOP matches for given search terms

      Calls fetch_OMOP_concepts on every item in search_terms.

      Parameters
      ----------
      search_terms: list[str]
          A list of queries to send to the OMOP database

      vocabulary_id: str
          An OMOP vocabulary_id to pass to the OMOP query to restrict the concepts received to a specific vocabulary

      concept_ancestor: str
          If 'y', then calls fetch_concept_ancestor()

      concept_relationship: str
          If 'y', then calls fetch_concept_relationship()

      concept_synonym: str
          If 'y', then queries the synonym table of the OMOP database for matches to the search terms

      search_threshold: int
          The threshold on fuzzy string matching for returned results

      max_separation_descendant: int
          The maximum separation to search for concept descendants

      max_separation_ancestor: int
          The maximum separation to search for concept ancestors

      Returns
      -------
      list
          A list of results for the search terms run with the other parameters provided.


   .. py:method:: fetch_OMOP_concepts(search_term: str, vocabulary_id: list | None, concept_ancestor: str, concept_relationship: str, concept_synonym: str, search_threshold: int, max_separation_descendant: int, max_separation_ancestor: int) -> list | None

      Fetch OMOP concepts for a given search term

      Runs queries against the OMOP database
      If concept_synonym != 'y', then a query is run that queries the concept table alone. If concept_synonym == 'y', then this search is expanded to the concept_synonym table.

      Any concepts returned by the query are then filtered by fuzzy string matching. Any concepts satisfying the concept threshold are returned.

      If the concept_ancestor and concept_relationship arguments are 'y', the relevant methods are called on these concepts and the result added to the output.

      Parameters
      ----------
      search_term: str
          A search term for a concept inserted into a query to the OMOP database
      vocabulary_id: list[str]
          A list of OMOP vocabularies to filter the findings by
      concept_ancestor: str
          If 'y' then appends the results of a call to fetch_concept_ancestor to the output
      concept_relationship: str
          If 'y' then appends the result of a call to fetch_concept_relationship to the output
      concept_synonym: str
          If 'y', checks the concept_synonym table for the search term
      search_threshold: int
          The threshold on fuzzy string matching for returned results
      max_separation_descendant: int
          The maximum separation to search for concept descendants

      max_separation_ancestor: int
          The maximum separation to search for concept ancestors


      Returns
      -------
      list | None
          A list of search results from the OMOP database if the query comes back with results, otherwise returns None


   .. py:method:: fetch_concept_ancestor(concept_id: str, max_separation_descendant: int, max_separation_ancestor: int)

      Fetch concept ancestor for a given concept_id

      Queries the OMOP database's ancestor table to find ancestors for the concept_id provided within the constraints of the degrees of separation provided.

      Parameters
      ----------
      concept_id: str
          The concept_id used to find ancestors
      max_separation_descendant: int
          The maximum level of separation allowed between descendant concepts and the provided concept
      max_separation_ancestor: int
          The maximum level of separation allowed between ancestor concepts and the provided concept

      Returns
      -------
      list
          A list of retrieved concepts and their relationships to the provided concept_id


   .. py:method:: fetch_concept_relationship(concept_id)

      Fetch concept relationship for a given concept_id

      Queries the concept_relationship table of the OMOP database to find the relationship between concepts

      Parameters
      ----------

      concept_id: str
          An id for a concept provided to the query for finding concept relationships

      Returns
      -------
      list
          A list of related concepts from the OMOP database



.. py:function:: run(opt: argparse.Namespace, search_term: str, logger: utils.logging_utils.Logger)

   Runs queries against the OMOP database

   Loads the query options from BaseOptions, then uses these to select which queries to run.

   Parameters
   ----------
   opt: argparse.Namespace
       Base options including the arguments relevant for OMOPMatcher methods
   search_term: str
       The name of a drug to use in queries to the OMOP database
   logger: Logger
       A logger for logging runs of the tool

   Returns
   -------
   list
       A list of OMOP concepts relating to the search term and relevant information


