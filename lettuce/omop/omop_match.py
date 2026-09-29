import re
from logging import Logger

from rapidfuzz import fuzz

from omop.db_manager import get_session
from omop.omop_queries import (
    query_ancestors_and_descendants_by_id,
    query_related_by_id,
    text_search_query,
)
from omop.preprocess import preprocess_search_term
from omop.result_models import (
    AncestorConcept,
    ConceptDescription,
    ConceptRelationship,
    ConceptResult,
    ConceptSynonym,
    RelatedConcept,
    SearchResult,
)


def calculate_similarity_score(concept_name, search_term) -> float:
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
    cleaned_concept_name = re.sub(r"\(.*?\)", "", concept_name).strip()
    score = fuzz.ratio(search_term.lower(), cleaned_concept_name.lower())
    return float(score)


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
        Whether or not to filter the query results based upon whether or not the search
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
        max_separation_ancestor: int = 1,
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

    def fetch_omop_concepts(self, search_term: str) -> list[ConceptResult] | None:
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
        list[ConceptResult] | None
            A list of search results from the OMOP database if the query comes back with results, otherwise returns None.
        """
        query = text_search_query(
            preprocess_search_term(search_term),
            self.vocabulary_id,
            self.standard_concept,
            self.concept_synonym,
        )

        seen_concept_ids = []
        results: list[ConceptResult] = []

        with get_session() as session:
            for row in session.execute(query).fetchall():
                # If we have not seen the concept_id for that row before because of synonyms, append the concept
                if row[0].concept_id not in seen_concept_ids:
                    concept = ConceptDescription.model_validate(row[0])
                    results.append(
                        ConceptResult(
                            concept=concept,
                            concept_name_similarity_score=calculate_similarity_score(
                                concept.concept_name, search_term
                            ),
                        )
                    )
                    seen_concept_ids.append(row[0].concept_id)

                # If there are synonyms, find the relevant concept and append the synonym
                if row[1] is not None:
                    result = next(
                        x for x in results if x.concept.concept_id == row[0].concept_id
                    )
                    result.concept_synonym.append(
                        ConceptSynonym(
                            concept_id=result.concept.concept_id,
                            concept_synonym_name=row[1],
                            concept_synonym_name_similarity_score=calculate_similarity_score(
                                row[1], search_term
                            ),
                        )
                    )

        results = [x for x in results if x.check_similarity(self.search_threshold)]

        if len(results) == 0:
            return None

        results.sort(key=lambda res: res.highest_similarity, reverse=True)

        if self.concept_ancestor:
            for result in results:
                result.concept_ancestor = self.fetch_concept_ancestors_and_descendants(
                    result.concept.concept_id
                )

        if self.concept_relationship:
            for result in results:
                result.concept_relationship = self.fetch_concept_relationships(
                    result.concept.concept_id
                )

        return results

    def fetch_concept_ancestors_and_descendants(
        self, concept_id: int
    ) -> list[AncestorConcept]:
        """
        Fetch concept ancestors and descendants for a given concept_id

        Queries the OMOP database's ancestor table to find ancestors and descendants for the concept_id provided within the constraints
        of the degrees of separation provided.

        Parameters
        ----------
        concept_id: int
            The concept_id used to find ancestors and descendants.

        Returns
        -------
        list[AncestorConcept]
            A list of retrieved concepts and their relationships to the provided concept_id
        """
        min_separation_ancestor = 1
        min_separation_descendant = 1

        query = query_ancestors_and_descendants_by_id(
            concept_id,
            min_separation_ancestor=min_separation_ancestor,
            max_separation_ancestor=self.max_separation_ancestor,
            min_separation_descendant=min_separation_descendant,
            max_separation_descendant=self.max_separation_descendant,
        )

        with get_session() as session:
            return [
                AncestorConcept.from_row_mapping(row)
                for row in session.execute(query).fetchall()
            ]

    def fetch_concept_relationships(self, concept_id: int) -> list[RelatedConcept]:
        """
        Fetch concept relationship for a given concept_id

        Queries the concept_relationship table of the OMOP database to find the relationship between concepts

        Parameters
        ----------
        concept_id: int
            An id for a concept provided to the query for finding concept relationships

        Returns
        -------
        list[RelatedConcept]
            A list of related concepts from the OMOP database
        """
        with get_session() as session:
            return [
                RelatedConcept(
                    concept=ConceptDescription.model_validate(row[0]),
                    relationship=ConceptRelationship(
                        concept_id_1=row[1], relationship_id=row[2], concept_id_2=row[3]
                    ),
                )
                for row in session.execute(query_related_by_id(concept_id)).fetchall()
            ]

    def run(self, search_terms: list[str]) -> list[SearchResult]:
        """
        Main method for the OMOPMatcherRunner class.

        Runs queries against the OMOP database for the user defined
        search terms and then performs fuzzy pattern matching on each one before selecting the best
        OMOP concept matches for each search term. Calls fetch_OMOP_concepts on every item in search_terms.

        Parameters
        ----------
        search_terms: list[str]
            The names of drugs to use in queries to the OMOP database

        Returns
        -------
        list[SearchResult]
            A list of OMOP concepts relating to the search term and relevant information
        """
        try:
            if not search_terms:
                self.logger.error("No valid search_term values provided")
                raise ValueError("No valid search_term values provided")

            self.logger.info(f"Calculating best OMOP matches for {search_terms}")
            overall_results = []

            for search_term in search_terms:
                omop_concepts = self.fetch_omop_concepts(search_term)
                overall_results.append(
                    SearchResult(search_term=search_term, concept=omop_concepts)
                )

            self.logger.info(f"Best OMOP matches for {search_terms} calculated")
            self.logger.info(
                f"OMOP Output: {[r.model_dump() for r in overall_results]}"
            )
            return overall_results

        except Exception as e:
            self.logger.error(f"Error in calculate_best_matches: {e}")
            raise ValueError(f"Error in calculate_best_OMOP_matches: {e}")
