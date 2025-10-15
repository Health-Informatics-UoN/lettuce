import marimo

__generated_with = "0.16.5"
app = marimo.App(
    width="medium",
    app_title="Lettuce",
    layout_file="layouts/ui.grid.json",
)


@app.cell
def _(mo):
    mo.md(r"""# Lettuce""")
    return


@app.cell
def _():
    from typing import List
    from dataclasses import dataclass
    import marimo as mo
    import polars as pl
    from sentence_transformers import SentenceTransformer

    from omop.db_manager import get_session
    from omop.omop_queries import (
        get_domains,
        get_vocabs,
        ts_rank_query,
        query_vector,
    )
    return (
        List,
        SentenceTransformer,
        dataclass,
        get_domains,
        get_session,
        get_vocabs,
        mo,
        pl,
        query_vector,
        ts_rank_query,
    )


@app.cell
def _(dataclass):
    @dataclass
    class ConceptSuggestion:
        concept_id: int
        concept_name: str
        domain_id: str
        vocabulary_id: str
        standard_concept: str
    return (ConceptSuggestion,)


@app.cell
def _(
    ConceptSuggestion,
    List,
    embed_model,
    get_session,
    query_vector,
    ts_rank_query,
):
    def search(
        search_term: str,
        domain: List[str] | None = None,
        vocabulary: List[str] | None = None,
        standard_concept: bool = True,
        valid_concept: bool = True,
        search_mode: str = "text-search",
    ) -> List[ConceptSuggestion]:
        if search_mode == "text-search":
            query = ts_rank_query(
                search_term=search_term,
                domain_id=domain,
                vocabulary_id=vocabulary,
                standard_concept=standard_concept,
                valid_concept=valid_concept,
                top_k=5,
            )
            with get_session() as session:
                results = session.execute(query).fetchall()
                return [
                    ConceptSuggestion(
                        concept_id=r.concept_id,
                        concept_name=r.concept_name,
                        domain_id=r.domain_id,
                        vocabulary_id=r.vocabulary_id,
                        standard_concept=r.standard_concept,
                    )
                    for r in results
                ]
        elif search_mode == "vector-search":
            query_embedding = embed_model.encode(search_term)
            query = query_vector(
                query_embedding=query_embedding,
                domain_id=domain,
                embed_vocab=vocabulary,
                standard_concept=standard_concept,
                valid_concept=valid_concept,
                describe_concept=True,
            )
            with get_session() as session:
                results = session.execute(query).fetchall()
                return [
                    ConceptSuggestion(
                        concept_id=r[0].concept_id,
                        concept_name=r[0].concept_name,
                        domain_id=r[0].domain_id,
                        vocabulary_id=r[0].vocabulary_id,
                        standard_concept=r[0].standard_concept,
                    )
                    for r in results
                ]
        else:
            # TODO: Implement ai-search
            raise NotImplementedError("I've not done the ai-search yet!")
    return (search,)


@app.cell
def _(mo):
    embeddings_model_name = mo.ui.text(value="BAAI/bge-small-en-v1.5")
    embeddings_enabled = mo.ui.checkbox(value=False)
    return embeddings_enabled, embeddings_model_name


@app.cell
def _(embeddings_enabled, embeddings_model_name, mo):
    config = mo.sidebar(
        [
            mo.md("## Configuration"),
            mo.md("### Embeddings"),
            mo.md("To use embeddings, you have to load an embeddings model"),
            mo.md(f"Embeddings model name {embeddings_model_name}"),
            mo.md(f"Enable embeddings? {embeddings_enabled}"),
            mo.md("### LLM"),
            mo.md(
                "To use the LLM, you must either have installed one of the llama-cpp extras or have an Ollama server running"
            ),
            mo.ui.dropdown(
                ["Ollama", "Llama.cpp"], value="Ollama", label="Inference type\n"
            ),
            mo.ui.text(value="http://localhost:11434", label="Ollama URL"),
            mo.ui.text(value="gemma3n:e4b", label="Model name"),
        ]
    )

    config
    return


@app.cell
def _(SentenceTransformer, embeddings_enabled, embeddings_model_name):
    if embeddings_enabled.value:
        embed_model = SentenceTransformer(embeddings_model_name.value)
    return (embed_model,)


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
    source_file = mo.ui.file(filetypes=[".csv"])
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
def _(domains, mo, search, source_df, source_term_column, vocabs):
    # Create arrays for each column of inputs
    source_terms = source_df[source_term_column.value].to_list()
    # TODO: initialise results with text-search
    get_preferences, set_preferences = mo.state(
        {
            "domain_dropdowns": mo.ui.array(
                [mo.ui.multiselect(domains) for _ in range(len(source_terms))]
            ),
            "vocabularies_dropdowns": mo.ui.array(
                [mo.ui.multiselect(vocabs) for _ in range(len(source_terms))]
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
                        ["text-search", "vector-search", "ai-search"],
                        value="text-search",
                    )
                    for _ in range(len(source_terms))
                ]
            ),
        }
    )

    get_results, set_results = mo.state(
        {i: search(v) for i, v in enumerate(source_terms)}
    )
    return get_preferences, get_results, set_results, source_terms


@app.cell
def _(get_results, search, set_results):
    # Modified search function that stores results
    def search_and_store(
        i,
        source_term,
        domain,
        vocabulary,
        standard_concept,
        valid_concept,
        search_mode,
    ):
        if domain == []:
            domain = None
        if vocabulary == []:
            vocabulary = None
        results = search(
            search_term=source_term,
            domain=domain,
            vocabulary=vocabulary,
            standard_concept=standard_concept,
            valid_concept=valid_concept,
            search_mode=search_mode,
        )
        current_results = get_results()
        current_results[i] = results
        set_results(current_results)
    return (search_and_store,)


@app.cell
def _(get_preferences, get_results, mo, search_and_store, source_terms):
    submit_buttons = mo.ui.array(
        [
            mo.ui.button(
                label="Submit",
                on_change=lambda v, i=i: search_and_store(
                    i,
                    source_terms[i],
                    get_preferences()["domain_dropdowns"][i].value,
                    get_preferences()["vocabularies_dropdowns"][i].value,
                    get_preferences()["standard_concept_checkboxes"][i].value,
                    get_preferences()["valid_concept_checkboxes"][i].value,
                    get_preferences()["search_modes"][i].value,
                ),
            )
            for i in range(len(source_terms))
        ]
    )
    # TODO: separate fields, dropdown for concept_id
    # TODO: persist selections with state
    # Show name as link to athena for concept
    view_suggestion = mo.ui.array(
        [
            mo.ui.dropdown(
                options={
                    f"{result.concept_name} ({result.concept_id})": result
                    for result in get_results().get(i, [])
                }
                if get_results().get(i)
                else {}
            )
            for i in range(len(source_terms))
        ]
    )
    # Create the table
    table = mo.ui.table(
        {
            "Source term": source_terms,
            "Domain": list(get_preferences()["domain_dropdowns"]),
            "Vocabulary": list(get_preferences()["vocabularies_dropdowns"]),
            "Standard Concept": list(
                get_preferences()["standard_concept_checkboxes"]
            ),
            "Valid Concept": list(get_preferences()["valid_concept_checkboxes"]),
            "Search mode": list(get_preferences()["search_modes"]),
            "Submit": list(submit_buttons),
            "Suggestion": list(view_suggestion),
        }
    )
    table
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
