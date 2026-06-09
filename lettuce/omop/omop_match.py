import re
from typing import List, Optional
from collections import defaultdict

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from logging import Logger
from omop.omop_queries import text_search_query, query_ancestors_and_descendants_by_id, query_related_by_id
from omop.db_manager import get_session 
from omop.preprocess import preprocess_search_term


class ConceptSynonym(BaseModel):
    """Model for concept synonym information"""
    concept_synonym_name: str
    concept_synonym_name_similarity_score: float


class ConceptRelationship(BaseModel):
    """Model for concept relationship information"""
    concept_id_1: int
    relationship_id: str
    concept_id_2: int


class AncestorRelationship(BaseModel):
    """Model for ancestor/descendant relationship information"""
    relationship_type: str
    ancestor_concept_id: int
    descendant_concept_id: int
    min_levels_of_separation: int
    max_levels_of_separation: int


class RelatedConcept(BaseModel):
    """Model for related concept with relationship details"""
    concept_name: str
    concept_id: int
    vocabulary_id: str
    concept_code: str
    relationship: ConceptRelationship


class AncestorConcept(BaseModel):
    """Model for ancestor/descendant concept with relationship details"""
    concept_name: str
    concept_id: int
    vocabulary_id: str
    concept_code: str
    relationship: AncestorRelationship


class OMOPConcept(BaseModel):
    """Model for OMOP concept search result"""
    concept_name: str
    concept_id: int
    vocabulary_id: str
    concept_code: str
    concept_name_similarity_score: float
    concept_synonym: List[ConceptSynonym] = Field(default_factory=list)
    concept_ancestor: List[AncestorConcept] = Field(default_factory=list)
    concept_relationship: List[RelatedConcept] = Field(default_factory=list)


class SearchResult(BaseModel):
    """Model for search term result"""
    search_term: str
    concept: Optional[List[OMOPConcept]]


class ConceptRow:
    """Internal data structure for processing concept query results"""
    def __init__(self, row_tuple):
        self.concept_id = row_tuple[0]
        self.concept_name = row_tuple[1]
        self.vocabulary_id = row_tuple[2]
        self.concept_code = row_tuple[3]
        self.concept_synonym_name = row_tuple[4] if len(row_tuple) > 4 else None
        self.concept_name_similarity_score = 0.0
        self.concept_synonym_name_similarity_score = 0.0


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
            return 0.0
        cleaned_concept_name = re.sub(r"\(.*?\)", "", concept_name).strip()
        score = fuzz.ratio(search_term.lower(), cleaned_concept_name.lower())
        return float(score)
            
    def fetch_omop_concepts(self, search_term: str) -> List[OMOPConcept] | None:
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
        List[OMOPConcept] | None
            A list of search results from the OMOP database if the query comes back with results, otherwise returns None. 
        """
        query = text_search_query(
            preprocess_search_term(search_term), self.vocabulary_id, self.standard_concept, self.concept_synonym
        )
        
        with get_session() as session:
            results = session.execute(query).fetchall() 
    
        if not results:  
            return None 
 
        # Convert results to ConceptRow objects and calculate similarity scores
        concept_rows = []
        for row_tuple in results:
            row = ConceptRow(row_tuple)
            row.concept_name_similarity_score = self.calculate_similarity_score(row.concept_name, search_term)
            row.concept_synonym_name_similarity_score = self.calculate_similarity_score(row.concept_synonym_name, search_term)
            concept_rows.append(row)

        # Filter by similarity threshold
        concept_ids_above_threshold = {
            row.concept_id for row in concept_rows
            if row.concept_name_similarity_score > self.search_threshold 
            or row.concept_synonym_name_similarity_score > self.search_threshold
        }
        
        if not concept_ids_above_threshold:
            return None

        filtered_rows = [row for row in concept_rows if row.concept_id in concept_ids_above_threshold]
        
        # Sort by highest similarity score
        filtered_rows.sort(
            key=lambda row: max(row.concept_name_similarity_score, row.concept_synonym_name_similarity_score),
            reverse=True
        )

        return self._format_concept_results(filtered_rows)

    def _format_concept_results(self, concept_rows: List[ConceptRow]) -> List[OMOPConcept]:
        """Format concept rows into Pydantic models"""
        # Group by concept_id
        grouped = defaultdict(list)
        for row in concept_rows:
            grouped[row.concept_id].append(row)

        formatted_results = []
        for _, rows in grouped.items():
            # Use first row for base concept info
            first_row = rows[0]
            
            # Collect synonyms
            synonyms = [
                ConceptSynonym(
                    concept_synonym_name=row.concept_synonym_name,
                    concept_synonym_name_similarity_score=row.concept_synonym_name_similarity_score
                )
                for row in rows if row.concept_synonym_name is not None
            ]
            
            # Create base concept
            concept = OMOPConcept(
                concept_name=first_row.concept_name,
                concept_id=first_row.concept_id,
                vocabulary_id=first_row.vocabulary_id,
                concept_code=first_row.concept_code,
                concept_name_similarity_score=first_row.concept_name_similarity_score,
                concept_synonym=synonyms
            )
            
            formatted_results.append(concept)

        # Fetch ancestors and relationships if requested
        if self.concept_ancestor:
            for concept in formatted_results:
                concept.concept_ancestor = self.fetch_concept_ancestors_and_descendants(concept.concept_id)
        
        if self.concept_relationship:
            for concept in formatted_results:
                concept.concept_relationship = self.fetch_concept_relationships(concept.concept_id)
                
        return formatted_results

    def fetch_concept_ancestors_and_descendants(self, concept_id: int) -> List[AncestorConcept]:
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
        List[AncestorConcept]
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

        return [
            AncestorConcept(
                concept_name=row[4],
                concept_id=row[1],
                vocabulary_id=row[5],
                concept_code=row[6],
                relationship=AncestorRelationship(
                    relationship_type=row[0],
                    ancestor_concept_id=row[2],
                    descendant_concept_id=row[3],
                    min_levels_of_separation=row[7],
                    max_levels_of_separation=row[8]
                )
            )
            for row in results
        ]

    def fetch_concept_relationships(self, concept_id: int) -> List[RelatedConcept]:
        """
        Fetch concept relationship for a given concept_id

        Queries the concept_relationship table of the OMOP database to find the relationship between concepts

        Parameters
        ----------
        concept_id: int
            An id for a concept provided to the query for finding concept relationships

        Returns
        -------
        List[RelatedConcept]
            A list of related concepts from the OMOP database
        """
        with get_session() as session: 
            query = query_related_by_id(concept_id)
            results = session.execute(query).fetchall()

        return [
            RelatedConcept(
                concept_name=row[4],
                concept_id=row[0],
                vocabulary_id=row[5],
                concept_code=row[6],
                relationship=ConceptRelationship(
                    concept_id_1=row[1],
                    relationship_id=row[2],
                    concept_id_2=row[3]
                )
            )
            for row in results
        ]

    def run(self, search_terms: List[str]) -> List[SearchResult]:
        """
        Main method for the OMOPMatcherRunner class. 
        
        Runs queries against the OMOP database for the user defined
        search terms and then performs fuzzy pattern matching on each one before selecting the best 
        OMOP concept matches for each search term. Calls fetch_OMOP_concepts on every item in search_terms.

        Parameters
        ----------
        search_terms: List[str]
            The names of drugs to use in queries to the OMOP database

        Returns
        -------
        List[SearchResult]
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
            self.logger.info(f"OMOP Output: {[r.model_dump() for r in overall_results]}")
            return overall_results

        except Exception as e:
            self.logger.error(f"Error in calculate_best_matches: {e}")
            raise ValueError(f"Error in calculate_best_OMOP_matches: {e}")
