from typing import List
import streamlit as st
import pandas as pd

from options.base_options import BaseOptions
from omop.db_manager import get_session
from omop.omop_queries import get_domains, get_vocabs, ts_rank_query
from streamlit import column_config

st.set_page_config(layout="wide")
settings = BaseOptions()

@st.cache_data
def fetch_domains():
    with get_session() as session:
        return [row[0] for row in session.execute(get_domains()).fetchall()][::-1]

@st.cache_data
def fetch_vocabs():
    with get_session() as session:
        return [row[0] for row in session.execute(get_vocabs()).fetchall()][::-1]

def batch_text_search(source_terms: List[str]):
    results = []
    for term in source_terms:
        with get_session() as session:
            result = session.execute(ts_rank_query(term, None, None, True, True, 1)).fetchall()
            if len(result) == 0:
                results.append([])
            else:
                results.append(result[0])
    return results

domains = fetch_domains()
vocabs = fetch_vocabs()
 
source_file = st.file_uploader("Choose a file containing source terms", type="csv")

if source_file is not None:
    source_df = pd.read_csv(source_file)
    st.dataframe(source_df.head())
    source_column = st.selectbox("Column containing source terms", [None, *source_df.columns])

    if source_column:
        initial_result = batch_text_search(source_df[source_column].tolist())
        result_df = pd.DataFrame(
                {
                    "source_term": source_df[source_column],
                    "domain": "Any",
                    "vocabulary": "Any",
                    "search": "text-search",
                    "accepted": False,
                    "result_id": [concept[1] if len(concept) != 0 else -1 for concept in initial_result],
                    "result_name": [concept[0] if len(concept) != 0 else "" for concept in initial_result],
                    }
                )

        st.data_editor(
                result_df,
                column_config={
                    "domain": st.column_config.SelectboxColumn(
                        "Domain ID",
                        help="The domain of the desired concept",
                        options=["Any", *domains]
                        ),
                    "vocabulary": st.column_config.SelectboxColumn(
                        "Vocabulary ID",
                        help="The vocabulary of the desired concept",
                        options=["Any", *vocabs]
                        ),
                    "search": st.column_config.SelectboxColumn(
                        "Search type",
                        help="Which version of search to use",
                        options=["text-search", "vector-search", "ai-search"]
                        ),
                    "accepted": st.column_config.CheckboxColumn("Accept Suggestion", help="Whether to accept this suggestion or try again")
                    }
                )

