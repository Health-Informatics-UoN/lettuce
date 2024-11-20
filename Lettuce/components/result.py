from typing import List, Dict, Any


class LettuceResult:
    def __init__(self, search_term: str) -> None:
        self.search_term = search_term

    def add_vector_search_results(self, vector_search_results: List[Dict[str, Any]]):
        self.vector_search_results = vector_search_results

    def add_llm_answer(self, llm_answer: str):
        self.llm_answer = llm_answer

    def get_query(self):
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
