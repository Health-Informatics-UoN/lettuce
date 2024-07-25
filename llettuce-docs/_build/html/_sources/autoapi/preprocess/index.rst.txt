preprocess
==========

.. py:module:: preprocess


Functions
---------

.. autoapisummary::

   preprocess.preprocess_search_term


Module Contents
---------------

.. py:function:: preprocess_search_term(term) -> str

   Preprocess a search term for use in a full-text search query.

   This function performs the following operations:

   1. Converts the input term to lowercase.
   2. Splits the term into individual words.
   3. Removes common stop words.
   4. Joins the remaining words with ' | ' for use in PostgreSQL's to_tsquery function.

   Args:
       term (str): The original search term.

   Returns:
       str: A preprocessed string ready for use in a full-text search query.

   Example:
       >>> preprocess_search_term("The quick brown fox")
       "quick | brown | fox"


