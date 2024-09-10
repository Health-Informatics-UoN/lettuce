import streamlit as st
import sseclient
import time
import requests
from app import PipelineOptions
from typing import List, Union


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
    names: List[str], use_llm: bool, final_api_call: bool, vocab_id: Union[str, None]
) -> sseclient.SSEClient:
    """
    This function makes an API call to the backend server to process the input names.

    Parameters
    ----------
    names: List[str]
        The list of names to process

    use_llm: bool
        Whether to use the LLM model for processing

    vocab_id: Union[str, None]
        The vocabulary ID to use for processing

    Returns
    -------
    sseclient.SSEClient
        The server-sent event client to stream the results
    """
    url = "http://127.0.0.1:8000/run"
    headers = {"Content-Type": "application/json"}
    pipe_opts = PipelineOptions(vocabulary_id=vocab_id)
    data = {
        "names": names,
        "pipeline_options": {
            "llm_model": pipe_opts.llm_model.value,
            "temperature": pipe_opts.temperature,
            "vocabulary_id": pipe_opts.vocabulary_id,
            "concept_ancestor": pipe_opts.concept_ancestor,
            "concept_relationship": pipe_opts.concept_relationship,
            "concept_synonym": pipe_opts.concept_synonym,
            "search_threshold": pipe_opts.search_threshold,
            "max_separation_descendants": pipe_opts.max_separation_descendants,
            "max_separation_ancestor": pipe_opts.max_separation_ancestor,
        },
        "use_llm": use_llm,
        "final_api_call": final_api_call,
    }
    print("Making API call with data:", data)
    response = requests.post(url, headers=headers, json=data, stream=True)
    return sseclient.SSEClient(response)
