import streamlit as st

csv_input_page = st.Page(
    page="csv_input.py", title="Input a csv", icon=":material/table:"
)
text_input_page = st.Page(
    page="text_input.py", title="Enter text manually", icon=":material/stylus:"
)
pg = st.navigation([text_input_page, csv_input_page])

pg.run()
