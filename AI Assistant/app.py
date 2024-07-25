import streamlit as st
from dotenv import load_dotenv

from chain.chains import Chains
from options.base_options import BaseOptions
from templates.html_templates import bot_template, css
from utils.utils import *


def run() -> None:
    """
    Run the streamlit app
    """
    load_dotenv()
    opt = BaseOptions().parse()
    informal_names_chunks = None
    chain = None

    st.set_page_config(page_title="BRC AI Assistant", page_icon="💊", layout="wide")
    st.write(css, unsafe_allow_html=True)

    if "upload_flag" not in st.session_state:
        st.session_state.upload_flag = False

    st.header("BRC AI Assistant")
    welcome_message(bot_template, opt.llm_model["model_name"])
    with st.sidebar:
        st.subheader("User Medications List")
        st.button(
            "i",
            key="info",
            help="The medications list should be in an excel file with the a column of 'informal_names'",
            type="secondary",
            disabled=True,
            use_container_width=False,
        )
        user_documents = st.file_uploader(
            "Upload your excel file", type=["xlsx", "xls"], accept_multiple_files=False
        )
        if st.button("Upload"):
            with st.spinner("Uploading"):
                informal_names_chunks = load_user_document(
                    user_documents, opt.df_chunk_size
                )
                if informal_names_chunks:
                    st.success("Uploaded successfully")
                    chain = Chains(
                        chain_type="conversion",
                        llm_model=opt.llm_model,
                        temperature=opt.temperature,
                        use_memory=opt.use_memory,
                        memory_input_key="informal_names",
                        use_simple_prompt=opt.use_simple_prompt,
                    ).get_chain()
                    st.session_state.upload_flag = True
                else:
                    st.error("Failed to upload")

    if st.session_state.upload_flag:
        with st.spinner("Processing"):
            conversion_histories, outputs = handle_conversion(
                informal_names_chunks,
                chain,
                use_memory=opt.use_memory,
                visualize_chunk=opt.visualize_chunk,
            )
            handle_output_df(
                outputs,
                visualize_chunk=opt.visualize_chunk,
                model_name=opt.llm_model["model_name"],
            )


if __name__ == "__main__":
    run()


# TODO
# 1. Handle the file with number of rows greater than LLM token limit
