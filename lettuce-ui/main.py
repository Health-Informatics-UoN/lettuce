import streamlit as st

from options.base_options import BaseOptions
from omop.db_manager import get_session
from omop.omop_queries import get_domains, ts_rank_query

settings = BaseOptions()

@st.cache_data
def fetch_domains():
    with get_session() as session:
        return [row[0] for row in session.execute(get_domains()).fetchall()][::-1]

domains = fetch_domains()
 
domain_select = st.selectbox("domains", domains)
