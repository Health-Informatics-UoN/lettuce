# UI utilities

## Functions
### `save_suggestions`
```python
def save_suggestions(
   filename: str,
   accepted_suggestion_fetcher: Callable,
   source_terms: List[str],
   ) -> None:
```
 Fetch the accepted suggestions and store them in a CSV file

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| `filename` | str | The filename to use to store the CSV file |
| `accepted_suggestion_fetcher` | Callable | A function to fetch the accepted suggestions to store |
| `source_terms` | List[str] | A list of source terms to match the accepted suggestions |


### `choose_result`
```python
choose_result(
   source_term_index: int,
   choice: ConceptSuggestion,
   suggestion_fetcher: Callable[[int], SuggestionRecord],
   accepted_updater: Callable[[int, AcceptedSuggestion], None],
   ) -> None:
```
Choose a concept suggestion and update the accepted suggestions

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| `source_term_index` | int | The index of the results to update |
| `choice_index` | int | The index of the choice in a suggestion to choose |
| `suggestion_fetcher` | `Callable[[int], SuggestionRecord]` | A function that fetches a SuggestionRecord at an index |
| `accepted_updater` | `Callable[[int, AcceptedSuggestion], None]` | A function that takes an index and an AcceptedSuggestion |

