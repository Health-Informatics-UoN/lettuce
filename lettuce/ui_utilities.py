import streamlit as st
import sseclient
import time
import requests
from options.pipeline_options import PipelineOptions


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


def make_api_call(
    names: list[str], skip_llm: bool, vocab_id: str | None
) -> sseclient.SSEClient:
    """
    Make a call to the Lettuce API to retrieve OMOP concepts.

    Parameters
    ----------
    names: list[str]
        The informal names to send to the API

    Returns
    -------
    sseclient.SSEClient
        The stream of events from the API
    """
    url = "http://127.0.0.1:8000/pipeline/"
    if not skip_llm:
        url = url + "db"
    headers = {"Content-Type": "application/json"}
    pipe_opts = PipelineOptions(vocabulary_id=vocab_id)
    data = {"names": names, "pipeline_options": pipe_opts.model_dump()}
    response = requests.post(url, headers=headers, json=data, stream=True)
    return sseclient.SSEClient(response)
