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

from logging import Logger
from omop.preprocess import preprocess_search_term


class OMOPMatcher:
    """
    This class retrieves matches from an OMOP database and returns the best
    """

    def __init__(self, logger: Logger):
        # Connect to database
        self.logger = logger
        load_dotenv()

        try:
            self.logger.info(
                "Initialize the PostgreSQL connection based on the environment variables"
            )
            DB_HOST = environ["DB_HOST"]
            DB_USER = environ["DB_USER"]
            DB_PASSWORD = quote_plus(environ["DB_PASSWORD"])
            DB_NAME = environ["DB_NAME"]
            DB_PORT = environ["DB_PORT"]
            DB_SCHEMA = environ["DB_SCHEMA"]

            connection_string = (
                f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
            )
            engine = create_engine(connection_string)
            logger.info(f"Connected to PostgreSQL database {DB_NAME} on {DB_HOST}")

        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise ValueError(f"Failed to connect to PostgreSQL: {e}")

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
        standard_concept: bool = True,
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
                    search_term=search_term,
                    vocabulary_id=vocabulary_id,
                    standard_concept=standard_concept,
                    concept_ancestor=concept_ancestor,
                    concept_relationship=concept_relationship,
                    concept_synonym=concept_synonym,
                    search_threshold=search_threshold,
                    max_separation_descendant=max_separation_descendant,
                    max_separation_ancestor=max_separation_ancestor,
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

    def fetch_OMOP_concepts(
        self,
        search_term: str,
        vocabulary_id: list | None,
        standard_concept: bool,
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
            search_term=preprocess_search_term(search_term),
            vocabulary_id=vocabulary_id,
            standard_concept=standard_concept,
            concept_synonym=concept_synonym,
        )
        Session = sessionmaker(self.engine)
        session = Session()
        results = session.execute(query).fetchall()
        results = pd.DataFrame(results)
        session.close()
        if not results.empty:
            # Define a function to calculate similarity score using the provided logic
            def calculate_similarity(row):
                if row is None:
                    return 0  # Return a default score (e.g., 0) for null values
                cleaned_concept_name = re.sub(r"\(.*?\)", "", row).strip()
                score = fuzz.ratio(search_term.lower(), cleaned_concept_name.lower())
                return score

            # Apply the score function to 'concept_name' and 'concept_synonym_name' columns
            results["concept_name_similarity_score"] = results["concept_name"].apply(
                calculate_similarity
            )
            results["concept_synonym_name_similarity_score"] = results[
                "concept_synonym_name"
            ].apply(calculate_similarity)

            concept_ids_above_threshold = set(
                results.loc[
                    (results["concept_name_similarity_score"] > search_threshold)
                    | (
                        results["concept_synonym_name_similarity_score"]
                        > search_threshold
                    ),
                    "concept_id",
                ]
            )

            # Step 2: Filter the original DataFrame to include all rows with these concept_ids
            results = results[results["concept_id"].isin(concept_ids_above_threshold)]

            # Sort the filtered results by the highest score (descending order)
            results = results.sort_values(
                by=[
                    "concept_name_similarity_score",
                    "concept_synonym_name_similarity_score",
                ],
                ascending=False,
            )

            # Group by 'concept_id' and aggregate the relevant columns
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

            # Define the final output format
            return [
                {
                    "concept_name": row["concept_name"],
                    "concept_id": row["concept_id"],
                    "vocabulary_id": row["vocabulary_id"],
                    "concept_code": row["concept_code"],
                    "concept_name_similarity_score": row[
                        "concept_name_similarity_score"
                    ],
                    "CONCEPT_SYNONYM": [
                        {
                            "concept_synonym_name": syn_name,
                            "concept_synonym_name_similarity_score": syn_score,
                        }
                        for syn_name, syn_score in zip(
                            row["concept_synonym_name"],
                            row["concept_synonym_name_similarity_score"],
                        )
                        if syn_name is not None
                    ],
                    "CONCEPT_ANCESTOR": (
                        self.fetch_concept_ancestor(
                            row["concept_id"],
                            max_separation_descendant,
                            max_separation_ancestor,
                        )
                        if concept_ancestor
                        else []
                    ),
                    "CONCEPT_RELATIONSHIP": (
                        self.fetch_concept_relationship(row["concept_id"])
                        if concept_relationship
                        else []
                    ),
                }
                for _, row in grouped_results.iterrows()
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
    search_term: List[str],
    logger: Logger,
    vocabulary_id: list[str],
    standard_concept: bool = True,
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
    omop_matcher = OMOPMatcher(logger)
    res = omop_matcher.calculate_best_matches(
        search_terms=search_term,
        vocabulary_id=vocabulary_id,
        standard_concept=standard_concept,
        concept_ancestor=concept_ancestor,
        concept_relationship=concept_relationship,
        concept_synonym=concept_synonym,
        search_threshold=search_threshold,
        max_separation_descendant=max_separation_descendant,
        max_separation_ancestor=max_separation_ancestor,
    )
    omop_matcher.close()
    return res
