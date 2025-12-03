from typing import List
from collections.abc import Callable
import polars as pl
from suggestions import AcceptedSuggestion, SuggestionRecord

def save_suggestions(
        filename: str,
        accepted_suggestion_fetcher: Callable,
        source_terms: List[str],
        ) -> None:
    """
    Fetch the accepted suggestions and store them in a CSV file

    Parameters
    ----------
    filename: str
        The filename to use to store the CSV file
    accepted_suggestion_fetcher: Callable
        A function to fetch the accepted suggestions to store
    source_terms: List[str]
        A list of source terms to match the accepted suggestions

    Returns
    -------
    None
    """
    accepted = [v for v in accepted_suggestion_fetcher().values()]
    save_dict = {
        "source_term": [],
        "search_term": [None for _ in range(len(source_terms))],
        "domains": [None for _ in range(len(source_terms))],
        "vocabs": [None for _ in range(len(source_terms))],
        "search_standard_concept": [None for _ in range(len(source_terms))],
        "valid_concept": [None for _ in range(len(source_terms))],
        "search_mode": [None for _ in range(len(source_terms))],
        "concept_id": [None for _ in range(len(source_terms))],
        "concept_name": [None for _ in range(len(source_terms))],
        "domain_id": [None for _ in range(len(source_terms))],
        "vocabulary_id": [None for _ in range(len(source_terms))],
        "standard_concept": [None for _ in range(len(source_terms))],
        "score": [None for _ in range(len(source_terms))],
    }
    for i in range(len(source_terms)):
        save_dict["source_term"].append(source_terms[i])
        if accepted[i] is not None:
            save_dict["search_term"][i] = accepted[i].search_term
            save_dict["domains"][i] = str(accepted[i].domains)
            save_dict["vocabs"][i] = str(accepted[i].vocabs)
            save_dict["search_standard_concept"][i] = accepted[i].search_standard_concept
            save_dict["valid_concept"][i] = accepted[i].valid_concept
            save_dict["search_mode"][i] = accepted[i].search_mode
            save_dict["concept_id"][i] = accepted[i].concept_id
            save_dict["concept_name"][i] = accepted[i].concept_name
            save_dict["domain_id"][i] = accepted[i].domain_id
            save_dict["vocabulary_id"][i] = accepted[i].vocabulary_id
            save_dict["standard_concept"][i] = accepted[i].standard_concept
            save_dict["score"][i] = accepted[i].score

    pl.DataFrame(save_dict).write_csv(filename)

def choose_result(
        source_term_index: int,
        choice_index: int,
        suggestion_fetcher: Callable[[int], SuggestionRecord],
        accepted_updater: Callable[[int, AcceptedSuggestion], None],
        ) -> None:
    """
    Choose a concept suggestion and update the accepted suggestions

    Parameters
    ----------
    source_term_index: int
        The index of the results to update
    choice_index: int
        The index of the choice in a suggestion to choose
    suggestion_fetcher: Callable[[int], SuggestionRecord]
        A function that fetches a SuggestionRecord at an index
    accepted_updater: Callable[[int, AcceptedSuggestion], None]
        A function that takes an index and an AcceptedSuggestion

    Returns
    -------
    None
    """
    suggestion = suggestion_fetcher(source_term_index)
    accepted_suggestion = suggestion.accept_suggestion(choice_index)
    accepted_updater(source_term_index, accepted_suggestion)
