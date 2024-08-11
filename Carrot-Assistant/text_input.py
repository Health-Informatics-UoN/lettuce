import json
from ui_utilities import (
    display_concept_info,
    stream_message,
    capitalize_words,
    make_api_call,
)
import sseclient
import streamlit as st


# ---> Process

#1. User enters informal names of medications.
#2. Send the informal names to the OMOP database.
#3. If no matches are found, ask the user if they want to try the LLM.
#4. If the user agrees, send the informal names to the LLM.
#5. Use the LLM-predicted names to query the OMOP database.
#6. Display the results to the user.


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

# Initialize session state

if "session_ended" not in st.session_state:
    st.session_state["session_ended"] = False

if st.session_state["session_ended"]:
    st.write("Session has ended. Thank you for using Carrot!")
    st.stop()

if st.button("Send"):
    if informal_names:
        names_list = [
            capitalize_words(name.strip()) for name in informal_names.split(",")
        ]
        with st.spinner("Processing..."):
            result_stream = make_api_call(names_list, use_llm=False, vocab_id=vocab_id)
            no_match_names = []

            for event in result_stream.events():
                response = json.loads(event.data)
                event_type = response["event"]
                message = response.get("message", "")

                if message:
                    stream_message(f"<p style='color: #4285F4;'>{message}</p>")

                # Display the results from the OMOP database

                if event_type == "omop_output":
                    for omop_data in response["data"]:
                        search_term = omop_data.get("search_term", "")
                        if not omop_data["CONCEPT"]:
                            stream_message(
                                f"<p style='color: #EA4335;'>No concepts found for {search_term}.</p>"
                            )
                            no_match_names.append(search_term)
                        else:
                            for i, concept in enumerate(omop_data["CONCEPT"], 1):
                                with st.expander(
                                    f"Concept {i}: {concept['concept_name']}",
                                    expanded=True,
                                ):
                                    display_concept_info(concept)

            if no_match_names:
                st.session_state["no_match_names"] = no_match_names
                st.session_state["vocab_id"] = vocab_id
                st.session_state["llm_requested"] = False

                st.rerun()

# Ask the user if they want to try the LLM if no matches are found in the OMOP database

if "no_match_names" in st.session_state and not st.session_state.get(
    "llm_requested", False
):
    user_choice = st.radio(
        "No match found in OMOP database. Would you like to try the LLM?",
        ["Select an option", "Yes", "No"],
        key="llm_option",
    )

    # If the user chooses "Yes", set the session state to request LLM predictions.
    # If the user chooses "No", end the session.

    if user_choice == "No":
        st.session_state["session_ended"] = True

    elif user_choice == "Yes":
        st.session_state["llm_requested"] = True
        st.rerun()

    if st.session_state["session_ended"]:
        if st.button("End Session"):
            st.write(
                "Thank you for using Carrot. Feel free to ask me more about informal names!"
            )
            st.stop()

# Process LLM predictions

if st.session_state.get("llm_requested", False):
    no_match_names = st.session_state["no_match_names"]
    vocab_id = st.session_state["vocab_id"]

    if "llm_processed_names" not in st.session_state:
        st.session_state["llm_processed_names"] = []

    llm_results = []

    # Processing LLM predictions

    with st.spinner("Processing with LLM..."):
        result_stream = make_api_call(no_match_names, use_llm=True, vocab_id=vocab_id)

        for event in result_stream.events():
            response = json.loads(event.data)
            event_type = response["event"]
            message = response.get("message", "")

            if message:
                stream_message(f"<p style='color: #4285F4;'>{message}</p>")

            if event_type == "llm_output":
                llm_output = response["data"]
                informal_name = llm_output["informal_name"]
                formal_name = llm_output["reply"]

                if informal_name not in st.session_state["llm_processed_names"]:
                    stream_message(
                        f'<p style="color: #34A853;">I found <b>{formal_name}</b> as the formal name for <b>{informal_name}</b></p>'
                    )
                    st.session_state["llm_processed_names"].append(informal_name)
                    llm_results.append(formal_name)

    # Start querying OMOP database with LLM-predicted names

    if llm_results:
        with st.spinner("Processing final OMOP query..."):
            new_result_stream = make_api_call(
                llm_results, use_llm=False, vocab_id=vocab_id
            )

            for new_event in new_result_stream.events():
                new_response = json.loads(new_event.data)
                new_event_type = new_response["event"]

                if new_event_type == "omop_output":
                    for new_omop_data in new_response["data"]:
                        if not new_omop_data["CONCEPT"]:
                            stream_message(
                                f"<p style='color: #EA4335;'>No concepts found for {new_omop_data['search_term']}.</p>"
                            )
                        else:
                            for i, concept in enumerate(new_omop_data["CONCEPT"], 1):
                                with st.expander(
                                    f"Concept {i}: {concept['concept_name']}",
                                    expanded=True,
                                ):
                                    display_concept_info(concept)

    # Stop any further API calls after processing

    st.session_state["session_ended"] = True
    st.stop()
