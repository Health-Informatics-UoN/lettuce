from typing import List, Dict, Any


class LettuceResult:
    """
    A class to hold the results of different stages of Lettuce pipelines

    Parameters
    ----------
    search_term: str
        The search or source term serving as input for the pipeline
    """

    def __init__(self, search_term: str) -> None:
        self.search_term = search_term

    def add_vector_search_results(self, vector_search_results: List[Dict[str, Any]]):
        self.vector_search_results = vector_search_results

    def add_llm_answer(self, llm_answer: str):
        self.llm_answer = llm_answer

    def get_query(self):
        """
        Insert the results of a vector search

        Parameters
        ----------
        vector_search_results: List[Dict[str, Any]]
            The results from running a vector database search from an embeddings object
        """
        self.vector_search_results = vector_search_results

    def add_llm_answer(self, llm_answer: str):
        """
        Insert the results of an LLM assistant's inference

        Parameters
        ----------
        llm_answer: str
            The reply of an LLM
        """
        self.llm_answer = llm_answer

    def get_query(self) -> str:
        """
        Retrieve the appropriate part of the result object for querying an OMOP-CDM database.
        If no previous stages have been executed, uses the search_term.
        If there's only a vector search result, uses the top result.
        If there is a response from an LLM, uses that

        Returns
        -------
        str
            Term for a database query
        """
        if hasattr(self, "llm_answer"):
            return self.llm_answer
        elif hasattr(self, "vector_search_results"):
            return self.vector_search_results[0]["content"]
        else:
            return self.search_term

    def add_matches(self, omop_matches: list, threshold: float):
        self.omop_fuzzy_threshold = threshold
        self.omop_matches = omop_matches

    def to_dict(self):
        """
        Inserts the matches retrieved from a database search, after fuzzy string filtering.

        Parameters
        ----------
        omop_matches: list
            A list of the matches retrieved from the database
        threshold: float
            The threshold used for filtering
        """
        self.omop_fuzzy_threshold = threshold
        self.omop_matches = omop_matches

    def to_dict(self) -> dict:
        """
        Serialises the result as a dictionary

        Returns
        -------
        dict
            Pipeline results serialised
        """
        out = dict()
        out["query"] = self.search_term
        if hasattr(self, "vector_search_results"):
            out["Vector Search Results"] = self.vector_search_results
        if hasattr(self, "llm_answer"):
            out["llm_answer"] = self.llm_answer
        if hasattr(self, "omop_matches"):
            out["OMOP fuzzy threshold"] = self.omop_fuzzy_threshold
            out["OMOP matches"] = self.omop_matches
        return out
