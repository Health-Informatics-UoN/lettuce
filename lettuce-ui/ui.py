import marimo

__generated_with = "0.16.5"
app = marimo.App(width="full", app_title="Lettuce")


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

    from omop.db_manager import get_session
    from omop.omop_queries import (
        get_domains,
        get_vocabs,
        ts_rank_query,
        query_vector,
        query_ids_matching_name,
    )
    from components.models import connect_to_ollama
    from components.pipeline import LLMPipeline
    from components.embeddings import EmbeddingModel, Embeddings
    from utils.logging_utils import logger
    return (
        Embeddings,
        LLMPipeline,
        List,
        connect_to_ollama,
        dataclass,
        get_domains,
        get_session,
        get_vocabs,
        logger,
        mo,
        pl,
        query_ids_matching_name,
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
        score: float = 0
    return (ConceptSuggestion,)


@app.cell
def _(
    ConceptSuggestion,
    Embeddings,
    LLMPipeline,
    List,
    connect_to_ollama,
    embeddings_model_name,
    get_session,
    llm_name,
    llm_url,
    logger,
    query_ids_matching_name,
    top_k,
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
                top_k=10,
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
            embedding_handler = Embeddings(
                model_name=embeddings_model_name.value,
                embed_vocab=vocabulary,
                domain_id=domain,
                standard_concept=standard_concept,
                valid_concept=valid_concept,
                top_k=10,
            )
            embedder = embedding_handler.get_embedder()
            embedding = embedder.run(search_term)
            retriever = embedding_handler.get_retriever()
            results = retriever.run(embedding["embedding"], describe_concept=True)
            print(results)
            return [
                ConceptSuggestion(
                    concept_id=r.Concept.concept_id,
                    concept_name=r.Concept.concept_name,
                    domain_id=r.Concept.domain_id,
                    vocabulary_id=r.Concept.vocabulary_id,
                    standard_concept=r.Concept.standard_concept,
                    score=r.score,
                )
                for r in results
            ]
        else:
            llm = connect_to_ollama(
                model_name=llm_name.value,
                url=llm_url.value,
                temperature=0.7,
                logger=logger,
            )
            assistant = LLMPipeline(
                llm=llm,
                temperature=0,
                logger=logger,
                embed_vocab=vocabulary,
                standard_concept=standard_concept,
            ).get_rag_assistant()
            answer = assistant.run(
                {
                    "prompt": {"informal_name": search_term, "domain": domain},
                    "query_embedder": {"text": search_term},
                },
            )
            reply = answer["llm"]["replies"][0].strip()
            query = query_ids_matching_name(
                query_concept=reply, vocabulary_ids=vocabulary, full_concept=True
            )
            with get_session() as session:
                results = session.execute(query).fetchall()
                return [
                    ConceptSuggestion(
                        concept_id=r[1],
                        concept_name=r[0],
                        domain_id=r[3],
                        vocabulary_id=r[4],
                        standard_concept=r[6],
                    )
                    for r in results
                ]
            if len(results) == 0:
                ts_query = ts_rank_query(
                    search_term=reply,
                    vocabulary_id=vocabulary,
                    domain_id=domain,
                    standard_concept=standard_concept,
                    valid_concept=valid_concept,
                    top_k=top_k,
                )
                with get_session() as session:
                    results = session.execute(ts_query).fetchall()
                return [
                    ConceptSuggestion(
                        concept_id=r.Concept.concept_id,
                        concept_name=r.Concept.concept_name,
                        domain_id=r.Concept.domain_id,
                        vocabulary_id=r.Concept.vocabulary_id,
                        standard_concept=r.Concept.standard_concept,
                    )
                    for r in results
                ]
    return (search,)


@app.cell
def _(mo):
    def update_search_options(search_options):
        if embeddings_enabled.value:
            search_options.add("vector-search")
        elif "vector-search" in search_options:
            search_options.remove("vector-search")


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
        from sentence_transformers import SentenceTransformer

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
def _(
    domains,
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
                        search_options,
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
def _(get_results, search, set_results):
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
def _(get_accepted, set_accepted):
    def choose_result(source_term_index, chosen_result):
        current_accepted = get_accepted()
        current_accepted[source_term_index] = chosen_result
        set_accepted(current_accepted)
    return (choose_result,)


@app.cell
def _(
    choose_result,
    get_accepted,
    get_preferences,
    get_results,
    mo,
    search_and_store,
    source_terms,
):
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
    # Show name as link to athena for concept
    view_suggestion = mo.ui.array(
        [
            mo.ui.dropdown(
                options={
                    result.concept_name: result
                    for result in get_results().get(i, [])
                }
                if get_results().get(i)
                else {},
                on_change=lambda v, i=i: choose_result(i, v),
            )
            for i in range(len(source_terms))
        ]
    )
    # Create the table
    search_table = mo.ui.table(
        {
            "Source term": source_terms,
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
                mo.md(concept.standard_concept) if concept is not None and concept.standard_concept is not None else ""
                for concept in get_accepted().values()
            ],
            "Score": [
                concept.score if concept is not None and concept.score is not None else ""
                for concept in get_accepted().values()
            ]
        }
    )
    return (suggestions_table,)


@app.cell
def _(mo, search_table, suggestions_table):
    mo.ui.tabs(
        {
            "Search concepts": search_table,
            "View suggestions": suggestions_table
        }
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
