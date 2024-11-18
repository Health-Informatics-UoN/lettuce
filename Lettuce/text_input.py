import json
from ui_utilities import display_concept_info, stream_message, capitalize_words, make_api_call
import sseclient
import streamlit as st

# Page configuration
st.set_page_config(page_title="Lettuce", page_icon="🥬", layout="wide")
st.markdown(
    "<h1 style='text-align: center; color: #34A853;'>Lettuce 🥬</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center;'>Hello, I'm Lettuce. Give me the name of a medication and I will tell you what Carrot knows about it!</p>",
    unsafe_allow_html=True,
)
informal_names = st.text_area(
    "Enter Informal names by (comma-separated) values. For example: (paracetamol, tylenol, omepra)",
    placeholder="Enter informal names of medications",
    key="input",
)

with st.expander("Search options"):
    skip_llm = st.checkbox("Ask the LLM first?", value=True)
    vocab_id = st.selectbox(label="Vocabulary ID", options=["RxNorm", "UK Biobank"])

if st.button("Send"):
    if informal_names:
        names_list = [
            capitalize_words(name.strip()) for name in informal_names.split(",")
        ]
        with st.spinner("Processing..."):
            result_stream: sseclient.SSEClient = make_api_call(names_list, skip_llm, vocab_id)

            # Stream the results
            for event in result_stream.events():
                response = json.loads(event.data)
                event_type = response["event"]

                # Stream the LLM output
                if event_type == "llm_output":
                    stream_message(
                        f'<p style="color: #34A853;">I found <b>{response["data"]["reply"]}</b> as the formal name for <b>{response["data"]["informal_name"]}</b></p>'
                    )

                # Stream the OMOP output
                elif event_type == "omop_output":
                    for j, omop_data in enumerate(response["data"]):
                        if (
                            omop_data["CONCEPT"] is None
                            or len(omop_data["CONCEPT"]) == 0
                        ):
                            stream_message(
                                f"<p style='color: #EA4335;'>No concepts found for {omop_data['search_term']}.</p>"
                            )
                        else:
                            for i, concept in enumerate(omop_data["CONCEPT"], 1):
                                with st.expander(
                                    f"Concept {i}: {concept['concept_name']}",
                                    expanded=True,
                                ):
                                    display_concept_info(concept)
    else:
        st.warning("Please enter an informal name before sending.")
