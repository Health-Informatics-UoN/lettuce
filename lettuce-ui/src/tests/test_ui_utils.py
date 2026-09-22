import pytest
import polars as pl
from suggestions import AcceptedSuggestion
from ui_utils import save_suggestions


class TestSaveSuggestions:
    """Tests for the save_suggestions function"""
    
    def test_save_with_all_accepted_suggestions(self, tmp_path):
        """Test saving when all source terms have accepted suggestions"""
        filename = tmp_path / "test_output.csv"
        source_terms = ["term1", "term2", "term3"]
        
        accepted_suggestions = {
            0: AcceptedSuggestion(
                search_term="search1",
                domains=["Domain1"],
                vocabs=["Vocab1"],
                search_standard_concept=True,
                valid_concept=True,
                search_mode="text-search",
                concept_id=123,
                concept_name="Concept 1",
                domain_id="Domain1",
                vocabulary_id="Vocab1",
                standard_concept="S",
                score=0.95
            ),
            1: AcceptedSuggestion(
                search_term="search2",
                domains=["Domain2", "Domain3"],
                vocabs=["Vocab2"],
                search_standard_concept=False,
                valid_concept=True,
                search_mode="vector-search",
                concept_id=456,
                concept_name="Concept 2",
                domain_id="Domain2",
                vocabulary_id="Vocab2",
                standard_concept="C",
                score=0.85
            ),
            2: AcceptedSuggestion(
                search_term="search3",
                domains=["Domain1"],
                vocabs=["Vocab1", "Vocab3"],
                search_standard_concept=True,
                valid_concept=False,
                search_mode="ai-search",
                concept_id=789,
                concept_name="Concept 3",
                domain_id="Domain1",
                vocabulary_id="Vocab3",
                standard_concept="S",
                score=0.75
            )
        }
        
        def fetcher():
            return accepted_suggestions
        
        save_suggestions(str(filename), fetcher, source_terms)
        
        assert filename.exists()
        df = pl.read_csv(filename)
        
        assert len(df) == 3
        assert df["source_term"].to_list() == source_terms
        assert df["search_term"].to_list() == ["search1", "search2", "search3"]
        assert df["concept_id"].to_list() == [123, 456, 789]
        assert df["concept_name"].to_list() == ["Concept 1", "Concept 2", "Concept 3"]
        assert df["score"].to_list() == [0.95, 0.85, 0.75]
    
    def test_save_with_none_accepted_suggestions(self, tmp_path):
        """Test saving when all accepted suggestions are None"""
        filename = tmp_path / "test_none.csv"
        source_terms = ["term1", "term2"]
        
        accepted_suggestions = {
            0: None,
            1: None
        }
        
        def fetcher():
            return accepted_suggestions
        
        save_suggestions(str(filename), fetcher, source_terms)
        
        assert filename.exists()
        df = pl.read_csv(filename)
        
        assert len(df) == 2
        assert df["source_term"].to_list() == source_terms
        # All other columns should be null
        assert all(df["search_term"].is_null())
        assert all(df["concept_id"].is_null())
        assert all(df["concept_name"].is_null())
    
    def test_save_with_mixed_accepted_suggestions(self, tmp_path):
        """Test saving with a mix of None and valid accepted suggestions"""
        filename = tmp_path / "test_mixed.csv"
        source_terms = ["term1", "term2", "term3"]
        
        accepted_suggestions = {
            0: AcceptedSuggestion(
                search_term="search1",
                domains=["Domain1"],
                vocabs=["Vocab1"],
                search_standard_concept=True,
                valid_concept=True,
                search_mode="text-search",
                concept_id=123,
                concept_name="Concept 1",
                domain_id="Domain1",
                vocabulary_id="Vocab1",
                standard_concept="S",
                score=0.95
            ),
            1: None,
            2: AcceptedSuggestion(
                search_term="search3",
                domains=["Domain3"],
                vocabs=["Vocab3"],
                search_standard_concept=False,
                valid_concept=True,
                search_mode="vector-search",
                concept_id=789,
                concept_name="Concept 3",
                domain_id="Domain3",
                vocabulary_id="Vocab3",
                standard_concept="C",
                score=0.80
            )
        }
        
        def fetcher():
            return accepted_suggestions
        
        save_suggestions(str(filename), fetcher, source_terms)
        
        assert filename.exists()
        df = pl.read_csv(filename)
        
        assert len(df) == 3
        assert df["source_term"].to_list() == source_terms
        assert df["search_term"][0] == "search1"
        assert df["search_term"][1] is None
        assert df["search_term"][2] == "search3"
        assert df["concept_id"][0] == 123
        assert df["concept_id"][1] is None
        assert df["concept_id"][2] == 789
    
    def test_save_with_empty_source_terms(self, tmp_path):
        """Test saving with empty source terms list"""
        filename = tmp_path / "test_empty.csv"
        source_terms = []
        
        def fetcher():
            return {}
        
        save_suggestions(str(filename), fetcher, source_terms)
        
        assert filename.exists()
        df = pl.read_csv(filename)
        
        assert len(df) == 0
        expected_columns = [
            "source_term", "search_term", "domains", "vocabs",
            "search_standard_concept", "valid_concept", "search_mode",
            "concept_id", "concept_name", "domain_id", "vocabulary_id",
            "standard_concept", "score"
        ]
        assert all(col in df.columns for col in expected_columns)
    
    def test_save_with_list_fields_converted_to_strings(self, tmp_path):
        """Test that list fields (domains, vocabs) are saved as strings"""
        filename = tmp_path / "test_lists.csv"
        source_terms = ["term1"]
        
        accepted_suggestions = {
            0: AcceptedSuggestion(
                search_term="search1",
                domains=["Domain1", "Domain2", "Domain3"],
                vocabs=["Vocab1", "Vocab2"],
                search_standard_concept=True,
                valid_concept=True,
                search_mode="text-search",
                concept_id=123,
                concept_name="Concept 1",
                domain_id="Domain1",
                vocabulary_id="Vocab1",
                standard_concept="S",
                score=0.95
            )
        }
        
        def fetcher():
            return accepted_suggestions
        
        save_suggestions(str(filename), fetcher, source_terms)
        
        df = pl.read_csv(filename)
        assert df["domains"][0] == str(["Domain1", "Domain2", "Domain3"])
        assert df["vocabs"][0] == str(["Vocab1", "Vocab2"])
