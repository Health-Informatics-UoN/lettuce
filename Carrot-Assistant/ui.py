import json
import time
from app import PipelineOptions

import requests
from requests.api import options
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
informal_name = st.text_input(
    "Informal name", placeholder="Enter the informal name of a medication", key="input"
)

with st.expander("Search options"):
    skip_llm = st.checkbox('Ask the LLM first?', value=True)
    vocab_id = st.selectbox(label='Vocabulary ID', options=["RxNorm", "UK Biobank"])
    llm_temperature = st.number_input(label='LLM temperature', min_value=0.0, max_value=3.0, step=0.1)
    


def stream_message(message: str) -> None:
    """
    Stream a message to the user, rendering HTML with a typewriter effect

    The message is displayed character by character. Streamlit's markdown functionality renders the message, allowing HTML formatting.

    Parameters
    ----------
    message: str
        The message to stream
    """
    t = st.empty()
    for i in range(len(message) + 1):
        t.markdown(message[:i], unsafe_allow_html=True)
        time.sleep(0.01)


def capitalize_words(s: str) -> str:
    """
    Capitalize each word in a string

    Parameters
    ----------
    s: str
        The string to capitalize

    Returns
    -------
    str
        The capitalized string
    """
    words = s.split()
    capitalized_words = [word[0].upper() + word[1:] if word else "" for word in words]
    return " ".join(capitalized_words)


def make_api_call(name: str) -> sseclient.SSEClient:
    """
    Make a call to the Llettuce API to retrieve OMOP concepts.

    Parameters
    ----------
    name: str
        The informal name to send to the API

    Returns
    -------
    sseclient.SSEClient
        The stream of events from the API
    """
    url = "http://127.0.0.1:8000/run"
    if not skip_llm:
        url = url+'_db'
    headers = {"Content-Type": "application/json"}
    pipe_opts = PipelineOptions(
        vocabulary_id=vocab_id,
        temperature=llm_temperature
    )
    data = {"name": name, "pipeline_options": pipe_opts.model_dump()}
    response = requests.post(url, headers=headers, json=data, stream=True)
    return sseclient.SSEClient(response)


def display_concept_info(concept: dict) -> None:
    """
    Display the concept information.
    An OMOP concept is formatted as HTML to be streamed to the user.

    Parameters
    ----------
    concept: dict
        The concept information
    """
    stream_message(
        f"<p style='color: #4285F4;'><b>Concept Name:</b> {concept['concept_name']}</p>"
    )
    stream_message(
        f"<p style='color: #4285F4;'><b>Concept ID:</b> {concept['concept_id']}</p>"
    )
    stream_message(
        f"<p style='color: #4285F4;'><b>Vocabulary ID:</b> {concept['vocabulary_id']}</p>"
    )
    stream_message(
        f"<p style='color: #4285F4;'><b>Concept Code:</b> {concept['concept_code']}</p>"
    )
    stream_message(
        f"<p style='color: #4285F4;'><b>Similarity Score:</b> {round(float(concept['concept_name_similarity_score']), 2)}</p>"
    )

    if concept["CONCEPT_SYNONYM"]:
        stream_message("<p style='color: #34A853;'><b>Synonyms:</b></p>")
        for synonym in concept["CONCEPT_SYNONYM"]:
            stream_message(f"<p style='color: #34A853;'>- {synonym}</p>")

    if concept["CONCEPT_ANCESTOR"]:
        stream_message("<p style='color: #FBBC05;'><b>Ancestors:</b></p>")
        for ancestor in concept["CONCEPT_ANCESTOR"]:
            stream_message(f"<p style='color: #FBBC05;'>- {ancestor}</p>")

    if concept["CONCEPT_RELATIONSHIP"]:
        stream_message("<p style='color: #EA4335;'><b>Relationships:</b></p>")
        for relationship in concept["CONCEPT_RELATIONSHIP"]:
            stream_message(f"<p style='color: #EA4335;'>- {relationship}</p>")


if st.button("Send"):
    if informal_name:
        with st.spinner("Processing..."):
            result_stream: sseclient.SSEClient = make_api_call(capitalize_words(informal_name))

            # Stream the results
            for event in result_stream.events():
                response = json.loads(event.data)
                event_type = response["event"]

                # Stream the LLM output
                if event_type == "llm_output":
                    stream_message(
                        f'<p style="color: #34A853;">I found <b>{response["data"]["reply"]}</b> as the formal name for <b>{informal_name}</b></p>'
                    )

                # Stream the OMOP output
                elif event_type == "omop_output":
                    for j, omop_data in enumerate(response["data"]):
                        if omop_data["CONCEPT"] is None or len(omop_data["CONCEPT"]) == 0:
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
