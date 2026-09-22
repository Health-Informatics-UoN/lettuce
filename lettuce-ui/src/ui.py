import marimo

__generated_with = "0.16.5"
app = marimo.App(width="full", app_title="Lettuce")


@app.cell
def _(mo):
    mo.md(r"""# Lettuce""")
    return


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from omop.db_manager import get_session
    from omop.omop_queries import (
        get_domains,
        get_vocabs,
    )
    from utils.logging_utils import logger
    from suggestions import SuggestionRecord, AcceptedSuggestion
    from search import search, search_and_store
    from ui_utils import choose_result, save_suggestions
    return (
        AcceptedSuggestion,
        SuggestionRecord,
        choose_result,
        get_domains,
        get_session,
        get_vocabs,
        logger,
        mo,
        pl,
        save_suggestions,
        search,
        search_and_store,
    )


@app.cell
def _(mo):
    embeddings_model_name = mo.ui.text(value="BGESMALL")
    embeddings_enabled = mo.ui.checkbox(value=False)

    llm_enabled = mo.ui.checkbox(value=False)
    llm_inference = mo.ui.dropdown(["Ollama", "Llama.cpp"], value="Ollama")
    llm_url = mo.ui.text(value="http://localhost:11434")
    llm_name = mo.ui.text(value="gemma3n:e4b")
    return (
        embeddings_enabled,
        embeddings_model_name,
        llm_enabled,
        llm_name,
        llm_url,
    )


@app.cell
def _(embeddings_enabled, llm_enabled):
    if embeddings_enabled.value:
        if llm_enabled.value:
            search_options = ["text-search", "vector-search", "ai-search"]
        else:
            search_options = ["text-search", "vector-search"]
    else:
        search_options = ["text-search"]
    return (search_options,)


@app.cell
def _(
    embeddings_enabled,
    embeddings_model_name,
    llm_enabled,
    llm_name,
    llm_url,
    mo,
):
    config = mo.sidebar(
        [
            mo.md("## Configuration"),
            mo.md("### Embeddings"),
            mo.md("To use embeddings, you have to load an embeddings model"),
            mo.md(f"Embeddings model name {embeddings_model_name}"),
            mo.md(f"Enable embeddings? {embeddings_enabled}"),
            mo.md("### LLM"),
            mo.md(
                "To use the LLM, you must either have installed one of the llama-cpp extras or have an Ollama server running, and to have enabled embeddings."
            ),
            mo.md(f"Enable LLM? {llm_enabled}"),
            mo.md(f"Ollama URL: {llm_url}"),
            mo.md(f"Model name: {llm_name}"),
        ]
    )

    config
    return


@app.cell
def _(get_domains, get_session, get_vocabs):
    with get_session() as session:
        domains = [row[0] for row in session.execute(get_domains()).fetchall()][
            ::-1
        ]
        vocabs = [row[0] for row in session.execute(get_vocabs()).fetchall()][::-1]
    return domains, vocabs


@app.cell
def _(mo):
    source_file = mo.ui.file(filetypes=[".csv"], label="Upload source file")
    source_file
    return (source_file,)


@app.cell
def _(mo, pl, source_file):
    source_df = pl.read_csv(source_file.value[0].contents)
    source_df_columns = source_df.columns
    source_term_column = mo.ui.dropdown(
        source_df_columns, label="Which column contains source terms?"
    )
    source_term_column if source_file.value else None
    return source_df, source_term_column


@app.cell
def _(domains, mo):
    default_domains = mo.ui.multiselect(domains, label="Default domains to search")
    default_domains
    return (default_domains,)


@app.cell
def _(mo, vocabs):
    default_vocabs = mo.ui.multiselect(
        vocabs, label="Default vocabularies to search"
    )
    default_vocabs
    return (default_vocabs,)


@app.cell
def _(
    SuggestionRecord,
    default_domains,
    default_vocabs,
    domains,
    embeddings_model_name,
    llm_name,
    llm_url,
    logger,
    mo,
    search,
    search_options,
    source_df,
    source_term_column,
    vocabs,
):
    # Create arrays for each column of inputs
    source_terms = source_df[source_term_column.value].to_list()
    get_preferences, set_preferences = mo.state(
        {
            "search_terms": mo.ui.array(
                [mo.ui.text(value=source_term) for source_term in source_terms]
            ),
            "domain_dropdowns": mo.ui.array(
                [
                    mo.ui.multiselect(domains, value=default_domains.value)
                    for _ in range(len(source_terms))
                ]
            ),
            "vocabularies_dropdowns": mo.ui.array(
                [
                    mo.ui.multiselect(vocabs, value=default_vocabs.value)
                    for _ in range(len(source_terms))
                ]
            ),
            "standard_concept_checkboxes": mo.ui.array(
                [mo.ui.checkbox(value=True) for _ in range(len(source_terms))]
            ),
            "valid_concept_checkboxes": mo.ui.array(
                [mo.ui.checkbox(value=True) for _ in range(len(source_terms))]
            ),
            "search_modes": mo.ui.array(
                [
                    mo.ui.dropdown(
                        search_options,
                        value="text-search",
                    )
                    for _ in range(len(source_terms))
                ]
            ),
        }
    )

    get_results, set_results = mo.state(
        {
            i: SuggestionRecord(
                search_term=v,
                domains=[],
                vocabs=[],
                standard_concept=True,
                valid_concept=True,
                search_mode="text-search",
                suggestion=search(
                    search_term=v,
                    domain=None,
                    vocabulary=None,
                    standard_concept=True,
                    valid_concept=True,
                    top_k=10,
                    search_mode="text-search",
                    embeddings_model_name=embeddings_model_name.value,
                    llm_name=llm_name.value,
                    llm_url=llm_url.value,
                    logger=logger,
                ),
            )
            for i, v in enumerate(source_terms)
        }
    )

    get_accepted, set_accepted = mo.state(
        {i: None for i in range(len(source_terms))}
    )
    return (
        get_accepted,
        get_preferences,
        get_results,
        set_accepted,
        set_results,
        source_terms,
    )


@app.cell
def _(SuggestionRecord, get_results, set_results):
    def update_results(result: SuggestionRecord, index: int) -> None:
        """
        Update the result at an index with the suggestions provided

        Parameters
        ----------
        result: SuggestionRecord
            The suggestions to store
        index: int
            The index of the results in which to store the suggestions
        """
        current_results = get_results()
        current_results[index] = result
        set_results(current_results)
    return (update_results,)


@app.cell
def _(AcceptedSuggestion, get_accepted, set_accepted):
    def update_accepted(
        index: int, accepted_suggestion: AcceptedSuggestion
    ) -> None:
        """
        Put an accepted suggestion into the accepted state

        Parameters
        ----------
        index: int
            The index of the accepted state to update
        accepted_suggestion: AcceptedSuggestion
            The AcceptedSuggestion to place at the provided index

        Returns
        -------
        None
        """
        current_accepted = get_accepted()
        current_accepted[index] = accepted_suggestion
        set_accepted(current_accepted)
    return (update_accepted,)


@app.cell
def _(
    choose_result,
    embeddings_model_name,
    get_accepted,
    get_preferences,
    get_results,
    llm_name,
    llm_url,
    logger,
    mo,
    search_and_store,
    source_terms,
    update_accepted,
    update_results,
):
    submit_buttons = mo.ui.array(
        [
            mo.ui.button(
                label="Submit",
                on_change=lambda v, i=i: search_and_store(
                    search_term=get_preferences()["search_terms"][i].value,
                    domain=get_preferences()["domain_dropdowns"][i].value,
                    vocabulary=get_preferences()["vocabularies_dropdowns"][
                        i
                    ].value,
                    standard_concept=get_preferences()[
                        "standard_concept_checkboxes"
                    ][i].value,
                    valid_concept=get_preferences()["valid_concept_checkboxes"][
                        i
                    ].value,
                    search_mode=get_preferences()["search_modes"][i].value,
                    embeddings_model_name=embeddings_model_name.value,
                    llm_name=llm_name.value,
                    llm_url=llm_url.value,
                    logger=logger,
                    top_k=10,
                    result_storer=lambda res: update_results(res, i),
                ),
            )
            for i in range(len(source_terms))
        ]
    )
    # Show name as link to athena for concept
    view_suggestion = mo.ui.array(
        [
            mo.ui.dropdown(
                options={
                    result.concept_name: result
                    for result in get_results().get(i, []).suggestion
                }
                if get_results().get(i)
                else {},
                on_change=lambda v, i=i: choose_result(
                    source_term_index=i,
                    choice=v,
                    suggestion_fetcher=lambda _: get_results()[i],
                    accepted_updater=update_accepted,
                ),
            )
            for i in range(len(source_terms))
        ]
    )
    # Create the table
    search_table = mo.ui.table(
        {
            "Source term": list(get_preferences()["search_terms"]),
            "Domain": list(get_preferences()["domain_dropdowns"]),
            "Vocabulary": list(get_preferences()["vocabularies_dropdowns"]),
            "Standard\nConcept": list(
                get_preferences()["standard_concept_checkboxes"]
            ),
            "Valid\nConcept": list(get_preferences()["valid_concept_checkboxes"]),
            "Search\nmode": list(get_preferences()["search_modes"]),
            "Submit": list(submit_buttons),
            "Suggestion": list(view_suggestion),
            "Concept ID": [
                mo.md(
                    f"[{concept.concept_id}](https://athena.ohdsi.org/search-terms/terms/{concept.concept_id})"
                )
                if concept is not None
                else ""
                for concept in get_accepted().values()
            ],
        }
    )
    return search_table, view_suggestion


@app.cell
def _(get_accepted, mo, source_terms, view_suggestion):
    suggestions_table = mo.ui.table(
        {
            "Source term": source_terms,
            "Suggestion": list(view_suggestion),
            "Concept ID": [
                mo.md(
                    f"[{concept.concept_id}](https://athena.ohdsi.org/search-terms/terms/{concept.concept_id})"
                )
                if concept is not None
                else ""
                for concept in get_accepted().values()
            ],
            "Concept Name": [
                mo.md(concept.concept_name) if concept is not None else ""
                for concept in get_accepted().values()
            ],
            "Domain ID": [
                mo.md(concept.domain_id) if concept is not None else ""
                for concept in get_accepted().values()
            ],
            "Vocabulary ID": [
                mo.md(concept.vocabulary_id) if concept is not None else ""
                for concept in get_accepted().values()
            ],
            "Standard Concept": [
                mo.md(concept.standard_concept)
                if concept is not None and concept.standard_concept is not None
                else ""
                for concept in get_accepted().values()
            ],
            "Score": [
                concept.score
                if concept is not None and concept.score is not None
                else ""
                for concept in get_accepted().values()
            ],
        }
    )
    return (suggestions_table,)


@app.cell
def _(mo, search_table, suggestions_table):
    mo.ui.tabs(
        {"Search concepts": search_table, "View suggestions": suggestions_table}
    )
    return


@app.cell
def _(mo):
    save_filename = mo.ui.text(
        label="Filename for results", value="search-results.csv"
    )
    return (save_filename,)


@app.cell
def _(get_accepted, mo, save_filename, save_suggestions, source_terms):
    save_button = mo.ui.button(
        label="Save results",
        on_click=save_suggestions(
            filename=save_filename.value,
            accepted_suggestion_fetcher=get_accepted,
            source_terms=source_terms,
        ),
    )
    return (save_button,)


@app.cell
def _(mo, save_button, save_filename):
    mo.md(f"""{save_filename}\t{save_button}""")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
