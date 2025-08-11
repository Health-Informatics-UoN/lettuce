import re
from typing import List

import pandas as pd
from rapidfuzz import fuzz
from omop.omop_queries import text_search_query, query_ancestors_and_descendants_by_id, query_related_by_id

from logging import Logger
from omop.db_manager import get_session 
from omop.preprocess import preprocess_search_term


class OMOPMatcher:
    """
    This class retrieves matches from an OMOP database and returns the best

    Parameters
    ----------
    logger: Logger
        Logging object. 

    vocabulary_id: list[str]
        A list of vocabularies to use for search

    concept_ancestor: bool
        Whether to return ancestor concepts in the result

    concept_relationship: bool
        Whether to return related concepts in the result
        
    concept_synonym: bool
        Whether to explore concept synonyms in the result
    
    standard_concept: bool 
        Whether or notto filter the query results based upon whether or not the search 
        space only includes standard concepts 

    search_threshold: int
        The fuzzy match threshold for results

    max_separation_descendant: int
        The maximum separation between a base concept and its descendants
        
    max_separation_ancestor: int
        The maximum separation between a base concept and its ancestors
    """

    def __init__(
        self, 
        logger: Logger, 
        vocabulary_id: list[str] | None,
        search_threshold: int = 80,
        concept_ancestor: bool = False,
        concept_relationship: bool = False,
        concept_synonym: bool = False,
        standard_concept: bool = False, 
        max_separation_descendant: int = 1,
        max_separation_ancestor: int = 1
    ):
        self.logger = logger
        self.vocabulary_id = vocabulary_id 
        self.search_threshold = search_threshold 
        self.concept_ancestor = concept_ancestor 
        self.concept_relationship = concept_relationship 
        self.concept_synonym = concept_synonym
        self.standard_concept = standard_concept 
        self.max_separation_descendant = max_separation_descendant
        self.max_separation_ancestor = max_separation_ancestor 

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
            
    def fetch_omop_concepts(self, search_term: str) -> list | None:
        """
        Fetch OMOP concepts for a given search term

        Runs queries against the OMOP database
        If concept_synonym != 'y', then a query is run that queries the concept table alone. If concept_synonym == 'y', then this search is expanded to the concept_synonym table.

        Any concepts returned by the query are then filtered by fuzzy string matching. Any concepts satisfying the concept threshold are returned.

        If the concept_ancestor and concept_relationship arguments are 'y', the relevant methods are called on these concepts and the result added to the output.

        Parameters
        ----------
        search_term: str
            A search term for a concept inserted into a query to the OMOP database.

        Returns
        -------
        list | None
            A list of search results from the OMOP database if the query comes back with results, otherwise returns None. 
        """
        query = text_search_query(
            preprocess_search_term(search_term), self.vocabulary_id, self.standard_concept, self.concept_synonym
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
            score_cols = ["concept_name_similarity_score", "concept_synonym_name_similarity_score"]  
        )

        return self._format_concept_results(results)
    
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
        score_cols: List[str],
        id_col: str = "concept_id" 
    ):
        concept_ids_above_threshold = set(
            results.loc[(results[score_cols] > self.search_threshold).any(axis=1), id_col]
        )
        results = results[results[id_col].isin(concept_ids_above_threshold)]
        results = results.sort_values(
            by=score_cols,
            ascending=False,
        )
        return results 

    def _format_concept_results(self, results: pd.DataFrame): 
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

        if self.concept_ancestor:
            for i, (_, row) in enumerate(grouped_results.iterrows()):
                formatted_results[i]["CONCEPT_ANCESTOR"] = self.fetch_concept_ancestors_and_descendants(row["concept_id"])
        
        if self.concept_relationship:
            for i, (_, row) in enumerate(grouped_results.iterrows()):
                formatted_results[i]["CONCEPT_RELATIONSHIP"] = self.fetch_concept_relationships(row["concept_id"])
                
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

    def fetch_concept_ancestors_and_descendants(self, concept_id: str):
        """
        Fetch concept ancestors and descendants for a given concept_id

        Queries the OMOP database's ancestor table to find ancestors and descendants for the concept_id provided within the constraints 
        of the degrees of separation provided.

        Parameters
        ----------
        concept_id: str
            The concept_id used to find ancestors and descendants.

        Returns
        -------
        list
            A list of retrieved concepts and their relationships to the provided concept_id
        """
        min_separation_ancestor = 1
        min_separation_descendant = 1

        query = query_ancestors_and_descendants_by_id(
            concept_id, 
            min_separation_ancestor=min_separation_ancestor, 
            max_separation_ancestor=self.max_separation_ancestor,   
            min_separation_descendant=min_separation_descendant, 
            max_separation_descendant=self.max_separation_descendant
        )
        
        with get_session() as session: 
            results = session.execute(query).fetchall()
            columns = ['relationship_type', 'concept_id', 'ancestor_concept_id', 
                       'descendant_concept_id', 'concept_name', 'vocabulary_id', 
                       'concept_code', 'min_levels_of_separation', 'max_levels_of_separation']
            results = pd.DataFrame(results, columns=columns)

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

    def fetch_concept_relationships(self, concept_id):
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
        with get_session() as session: 
            query = query_related_by_id(concept_id)
            results = session.execute(query).fetchall()
            columns = [
                'concept_id', 'concept_id_1', 'relationship_id', 'concept_id_2',
                'concept_name', 'vocabulary_id', 'concept_code'
            ]
            results = pd.DataFrame(results, columns=columns)

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

    def run(self, search_terms: List[str]):
        """
        Main method for the OMOPMatcherRunner class. 
        
        Runs queries against the OMOP database for the user defined
        search terms and then performs fuzzy pattern matching on each one before selecting the best 
        OMOP concept matches for each search term. Calls fetch_OMOP_concepts on every item in search_terms.

        Parameters
        ----------
        search_terms: str
            The name of a drug to use in queries to the OMOP database

        Returns
        -------
        list
            A list of OMOP concepts relating to the search term and relevant information
        """
        try:
            if not search_terms:
                self.logger.error("No valid search_term values provided")
                raise ValueError("No valid search_term values provided")

            self.logger.info(f"Calculating best OMOP matches for {search_terms}")
            overall_results = []

            for search_term in search_terms:
                OMOP_concepts = self.fetch_omop_concepts(search_term)

                overall_results.append(
                    {"search_term": search_term, "CONCEPT": OMOP_concepts}
                )

            self.logger.info(f"Best OMOP matches for {search_terms} calculated")
            self.logger.info(f"OMOP Output: {overall_results}")
            return overall_results

        except Exception as e:
            self.logger.error(f"Error in calculate_best_matches: {e}")
            raise ValueError(f"Error in calculate_best_OMOP_matches: {e}")
