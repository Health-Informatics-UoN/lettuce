import re
from os import environ
from urllib.parse import quote_plus
from typing import List

import pandas as pd
from dotenv import load_dotenv
from rapidfuzz import fuzz
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from omop.omop_queries import text_search_query
from omop.db_manager import get_session, DB_SCHEMA, engine 

from logging import Logger
from omop.preprocess import preprocess_search_term


class OMOPMatcher:
    """
    This class retrieves matches from an OMOP database and returns the best
    """

    def __init__(self, logger: Logger):
        # Connect to database
        self.logger = logger
        self.engine = engine
        self.schema = DB_SCHEMA

    def close(self):
        """Close the engine connection."""
        self.engine.dispose()
        self.logger.info("PostgreSQL connection closed.")

    def calculate_best_matches(
        self,
        search_terms: List[str],
        vocabulary_id: list | None = None,
        concept_ancestor: bool = True,
        concept_relationship: bool = True,
        concept_synonym: bool = True,
        search_threshold: int = 0,
        max_separation_descendant: int = 1,
        max_separation_ancestor: int = 1,
    ) -> list:
        # As all this does is call fetch_OMOP_concepts, maybe everything but search_terms should be put into kwargs

        """
        Calculate best OMOP matches for given search terms

        Calls fetch_OMOP_concepts on every item in search_terms.

        Parameters
        ----------
        search_terms: List[str]
            A list of queries to send to the OMOP database

        vocabulary_id: str
            An OMOP vocabulary_id to pass to the OMOP query to restrict the concepts received to a specific vocabulary

        concept_ancestor: bool
            If 'y', then calls fetch_concept_ancestor()

        concept_relationship: bool
            If 'y', then calls fetch_concept_relationship()

        concept_synonym: bool
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
        """
        try:
            if not search_terms:
                self.logger.error("No valid search_term values provided")
                raise ValueError("No valid search_term values provided")

            self.logger.info(f"Calculating best OMOP matches for {search_terms}")
            overall_results = []

            for search_term in search_terms:
                OMOP_concepts = self.fetch_OMOP_concepts(
                    search_term,
                    vocabulary_id,
                    concept_ancestor,
                    concept_relationship,
                    concept_synonym,
                    search_threshold,
                    max_separation_descendant,
                    max_separation_ancestor,
                )

                overall_results.append(
                    {"search_term": search_term, "CONCEPT": OMOP_concepts}
                )

            self.logger.info(f"Best OMOP matches for {search_terms} calculated")
            self.logger.info(f"OMOP Output: {overall_results}")
            return overall_results

        except Exception as e:
            self.logger.error(f"Error in calculate_best_matches: {e}")
            raise ValueError(f"Error in calculate_best_OMOP_matches: {e}")

    @staticmethod 
    def calculate_similarity_score(concept_name, search_term):
        """
        Calculates a fuzzy similarity score between a concept name and a search term.

        This method is designed to compare drug concept names, such as those found in OMOP 
        vocabularies, with user-entered search terms. The concept name is cleaned by 
        removing all content inside parentheses () before comparison and both strings 
        are lowercased before comparison to ensure case-insensitive matching.

        Parameters
        ----------
        concept_name (str): 
            The full OMOP drug concept name to be compared. 

        search_term (str): 
            The user-entered term to compare against, such as "paracetamol" or "acetaminophen".
        
        Returns
        -------
            float: A similarity score between 0 and 100, where higher values indicate a stronger match.
        """
        if concept_name is None:
            return 0  # Return a default score (e.g., 0) for null values
        cleaned_concept_name = re.sub(r"\(.*?\)", "", concept_name).strip()
        score = fuzz.ratio(search_term.lower(), cleaned_concept_name.lower())
        return score
            
    def fetch_OMOP_concepts(
        self,
        search_term: str,
        vocabulary_id: list | None,
        concept_ancestor: bool,
        concept_relationship: bool,
        concept_synonym: bool,
        search_threshold: int,
        max_separation_descendant: int,
        max_separation_ancestor: int,
    ) -> list | None:
        """
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
        """
        query = text_search_query(
            preprocess_search_term(search_term), vocabulary_id, concept_synonym
        )
        
        with get_session() as session:
           results = session.execute(query).fetchall()
           results = pd.DataFrame(results) 
        
        if results.empty:  
            return None 
 
        # Apply the score function to 'concept_name' and 'concept_synonym_name' columns
        results = self._apply_concept_similarity_score_to_columns(
            results, 
            search_term, 
            source_cols=["concept_name", "concept_synonym_name"],
            target_cols=["concept_name_similarity_score", "concept_synonym_name_similarity_score"]
        )

        # Filter the original DataFrame to include all rows with these concept_ids
        # Sort the filtered results by the highest score (descending order)
        results = self._filter_and_sort_by_concept_ids_above_similarity_score_threshold(
            results, 
            search_threshold, 
            score_cols = ["concept_name_similarity_score", "concept_synonym_name_similarity_score"]  
        )

        return self._format_concept_results(
            results, 
            max_separation_descendant, 
            max_separation_ancestor, 
            concept_ancestor, 
            concept_relationship
        )
    
    def _apply_concept_similarity_score_to_columns(
            self, 
            results:  pd.DataFrame, 
            search_term: str, 
            source_cols: List[str], 
            target_cols: List[str]
    ): 
        for col_src, col_target in zip(source_cols, target_cols): 
            results[col_target] = results[col_src].apply(
                lambda row: OMOPMatcher.calculate_similarity_score(row, search_term)
            ) 
        return results 
    
    def _filter_and_sort_by_concept_ids_above_similarity_score_threshold(
        self, 
        results: pd.DataFrame,
        threshold: float,
        score_cols: List[str],
        id_col: str = "concept_id" 
    ):
        concept_ids_above_threshold = set(results.loc[(results[score_cols] > threshold).any(axis=1), id_col])
        results = results[results[id_col].isin(concept_ids_above_threshold)]
        results = results.sort_values(
            by=score_cols,
            ascending=False,
        )
        return results 

    def _format_concept_results(
        self, 
        results: pd.DataFrame, 
        max_separation_descendant: int,
        max_separation_ancestor: int,
        concept_ancestor: bool = False,
        concept_relationship: bool = False
    ): 
        grouped_results = (
            results.groupby("concept_id")
            .agg(
                {
                    "concept_name": "first",
                    "vocabulary_id": "first",
                    "concept_code": "first",
                    "concept_name_similarity_score": "first",
                    "concept_synonym_name": list,
                    "concept_synonym_name_similarity_score": list,
                }
            )
            .reset_index()
        )

        formatted_results = []
        for _, row in grouped_results.iterrows():
            result = self._format_base_concept(row)
            result["CONCEPT_SYNONYM"] = self._format_concept_synonyms(row)
            result["CONCEPT_ANCESTOR"] = []  
            result["CONCEPT_RELATIONSHIP"] = []  
            formatted_results.append(result)

        if concept_ancestor:
            for i, (_, row) in enumerate(grouped_results.iterrows()):
                formatted_results[i]["CONCEPT_ANCESTOR"] = self.fetch_concept_ancestor(
                    row["concept_id"],
                    max_separation_descendant,
                    max_separation_ancestor
                )
        
        if concept_relationship:
            for i, (_, row) in enumerate(grouped_results.iterrows()):
                formatted_results[i]["CONCEPT_RELATIONSHIP"] = self.fetch_concept_relationship(
                    row["concept_id"]
            )
                
        return formatted_results
    
    def _format_base_concept(self, row):
        """Format the base concept information from a row."""
        return {
            "concept_name": row["concept_name"],
            "concept_id": row["concept_id"],
            "vocabulary_id": row["vocabulary_id"],
            "concept_code": row["concept_code"],
            "concept_name_similarity_score": row["concept_name_similarity_score"]
        }

    def _format_concept_synonyms(self, row):
        """Format the concept synonyms from a row."""
        return [
            {
                "concept_synonym_name": syn_name,
                "concept_synonym_name_similarity_score": syn_score,
            }
            for syn_name, syn_score in zip(
                row["concept_synonym_name"],
                row["concept_synonym_name_similarity_score"],
            )
            if syn_name is not None
        ]

    def fetch_concept_ancestor(
        self,
        concept_id: str,
        max_separation_descendant: int,
        max_separation_ancestor: int,
    ):
        """
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
        """

        query = f"""
            (
                SELECT
                    'Ancestor' as relationship_type,
                    ca.ancestor_concept_id AS concept_id,
                    ca.ancestor_concept_id,
                    ca.descendant_concept_id,
                    c.concept_name,
                    c.vocabulary_id,
                    c.concept_code,
                    ca.min_levels_of_separation,
                    ca.max_levels_of_separation
                FROM
                    {self.schema}.concept_ancestor ca
                JOIN
                    {self.schema}.concept c ON ca.ancestor_concept_id = c.concept_id
                WHERE
                    ca.descendant_concept_id = %s AND
                    ca.min_levels_of_separation >= %s AND
                    ca.max_levels_of_separation <= %s
            )
            UNION
            (
                SELECT
                    'Descendant' as relationship_type,
                    ca.descendant_concept_id AS concept_id,
                    ca.ancestor_concept_id,
                    ca.descendant_concept_id,
                    c.concept_name,
                    c.vocabulary_id,
                    c.concept_code,
                    ca.min_levels_of_separation,
                    ca.max_levels_of_separation
                FROM
                    {self.schema}.concept_ancestor ca
                JOIN
                    {self.schema}.concept c ON ca.descendant_concept_id = c.concept_id
                WHERE
                    ca.ancestor_concept_id = %s AND
                    ca.min_levels_of_separation >= %s AND
                    ca.max_levels_of_separation <= %s
            )
        """
        min_separation_ancestor = 1
        min_separation_descendant = 1

        params = (
            concept_id,
            min_separation_ancestor,
            max_separation_ancestor,
            concept_id,
            min_separation_descendant,
            max_separation_descendant,
        )

        results = (
            pd.read_sql(query, con=self.engine, params=params)
            .drop_duplicates()
            .query("concept_id != @concept_id")
        )

        return [
            {
                "concept_name": row["concept_name"],
                "concept_id": row["concept_id"],
                "vocabulary_id": row["vocabulary_id"],
                "concept_code": row["concept_code"],
                "relationship": {
                    "relationship_type": row["relationship_type"],
                    "ancestor_concept_id": row["ancestor_concept_id"],
                    "descendant_concept_id": row["descendant_concept_id"],
                    "min_levels_of_separation": row["min_levels_of_separation"],
                    "max_levels_of_separation": row["max_levels_of_separation"],
                },
            }
            for _, row in results.iterrows()
        ]

    def fetch_concept_relationship(self, concept_id):
        """
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
        """
        query = f"""
            SELECT
                cr.concept_id_2 AS concept_id,
                cr.concept_id_1,
                cr.relationship_id,
                cr.concept_id_2,
                c.concept_name,
                c.vocabulary_id,
                c.concept_code
            FROM
                {self.schema}.concept_relationship cr
            JOIN
                {self.schema}.concept c ON cr.concept_id_2 = c.concept_id
            WHERE
                cr.concept_id_1 = %s AND
                cr.valid_end_date > NOW()
        """
        results = (
            pd.read_sql(query, con=self.engine, params=(concept_id,))
            .drop_duplicates()
            .query("concept_id != @concept_id")
        )

        return [
            {
                "concept_name": row["concept_name"],
                "concept_id": row["concept_id"],
                "vocabulary_id": row["vocabulary_id"],
                "concept_code": row["concept_code"],
                "relationship": {
                    "concept_id_1": row["concept_id_1"],
                    "relationship_id": row["relationship_id"],
                    "concept_id_2": row["concept_id_2"],
                },
            }
            for _, row in results.iterrows()
        ]

    def run(
        self, 
        search_term: List[str],
        vocabulary_id: list[str],
        search_threshold: int = 80,
        concept_ancestor: bool = False,
        concept_relationship: bool = False,
        concept_synonym: bool = False,
        max_separation_descendant: int = 1,
        max_separation_ancestor: int = 1,
    ):
        """
        Runs queries against the OMOP database

        Loads the query options from BaseOptions, then uses these to select which queries to run.

        Parameters
        ----------
        vocabulary_id: list[str]
            A list of vocabularies to use for search
        concept_ancestor: bool
            Whether to return ancestor concepts in the result
        concept_relationship: bool
            Whether to return related concepts in the result
        concept_synonym: bool
            Whether to explore concept synonyms in the result
        search_threshold: int
            The fuzzy match threshold for results
        max_separation_descendant: int
            The maximum separation between a base concept and its descendants
        max_separation_ancestor: int
            The maximum separation between a base concept and its ancestors
        search_term: str
            The name of a drug to use in queries to the OMOP database
        logger: Logger
            A logger for logging runs of the tool

        Returns
        -------
        list
            A list of OMOP concepts relating to the search term and relevant information
        """
        res = self.calculate_best_matches(
            search_term,
            vocabulary_id,
            concept_ancestor,
            concept_relationship,
            concept_synonym,
            search_threshold,
            max_separation_descendant,
            max_separation_ancestor,
        )
        return res
