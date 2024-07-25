import ast
import time
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader

from chain.chains import Chains
from constant import PACKAGE_ROOT_PATH
from templates.html_templates import bot_template


def load_user_document(file: str, df_chunk_size: int = 100) -> List[Document]:
    """
    Load the user document

    Parameters:
    ----------
    file: str
        The excel file path to load
    df_chunk_size: int
        The chunk size to use

    Returns:
    -------
    List[Document]
        The list of documents
    """

    df = pd.read_excel(file, engine="openpyxl")
    df_columns = df.columns
    if len(df_columns) > 1:
        df = df.iloc[:, 0].to_frame()
    df.columns = ["informal_names"]

    df = df.dropna(subset=["informal_names"])
    df = df[df["informal_names"].str.len() > 1]
    chunked_dfs = [
        df[i : i + df_chunk_size] for i in range(0, df.shape[0], df_chunk_size)
    ]
    chunks = []
    for chunk in chunked_dfs:
        loader = DataFrameLoader(chunk, page_content_column="informal_names")
        docs = loader.load()
        chunks.append(docs)

    return chunks


def split_docs(
    docs: List[Document], chunk_size: int = 2000, chunk_overlap: int = 200
) -> List[Document]:
    """
    Split the documents

    Parameters:
    ----------
    docs: List[Document]
        The list of documents
    chunk_size: int
        The chunk size to use
    chunk_overlap: int
        The chunk overlap to use

    Returns:
    -------
    List[Document]
        The list of documents
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    chunks = text_splitter.split_documents(docs)
    return chunks


@st.cache_data(show_spinner=False)
def welcome_message(bot_template: str, model_name: str) -> None:
    """
    The welcome message for the bot

    Parameters:
    ----------
    bot_template: str
        The bot template
    model_name: str
        The model name
    """
    import time

    message = f"Welcome to the BRC AI Assistant. Please upload your excel file containing the list of informal medications names.\n Using {model_name} model."
    t = st.empty()
    for i in range(len(message) + 1):
        t.markdown(bot_template.replace("{{MSG}}", message[:i]), unsafe_allow_html=True)
        time.sleep(0.02)


def handle_conversion(
    informal_names_chunks: List[List[Document]],
    conversion_chain: Chains,
    use_memory: bool = False,
    visualize_chunk: bool = False,
):
    """
    Handle the conversion

    Parameters:
    ----------
    informal_names_chunks: List[List[Document]]
        The list of informal names chunks
    conversion_chain: Chains
        The conversion chain
    use_memory: bool
        Whether to use memory
    visualize_chunk: bool
        Whether to visualize the chunk
    """
    tick = time.time()
    informal_names = []
    outputs = []
    conversion_histories = []
    for i, informal_names in enumerate(informal_names_chunks):
        print(f"Processing chunk {i+1} of {len(informal_names_chunks)}")
        res = conversion_chain.invoke(
            {
                "informal_names": informal_names,
                "informal_names_length": len(informal_names),
            }
        )
        if use_memory:
            conversion_history = res["chat_history"]
            AI_Assistant_Output = conversion_history[-1].content
            conversion_histories.append(conversion_history)
            if visualize_chunk:
                visualize_output(
                    AI_Assistant_Output
                )  # To visualize the output one by one
            outputs.append(AI_Assistant_Output)
        else:
            AI_Assistant_Output = res["text"]
            if visualize_chunk:
                visualize_output(
                    AI_Assistant_Output
                )  # To visualize the output one by one
            outputs.append(AI_Assistant_Output)

    tock = time.time()
    print(f"Total time taken: {(tock-tick)/60:.2f} minutes")
    return conversion_histories, outputs


def visualize_output(output: str) -> None:
    """
    Visualize the output

    Parameters:
    ----------
    output: str
        The output
    """
    try:
        output = eval(output)
        df = pd.DataFrame(output)
        st.dataframe(df, height=600, use_container_width=True)

    except Exception as e:
        print(e)
        try:
            df = handle_corrupted_data(output)
            st.dataframe(df, height=600, use_container_width=True)
        except Exception as e:
            print(e)
            output = str(output)
            t = st.empty()
            for i in range(len(output) + 1):
                t.markdown(output[:i], unsafe_allow_html=True)
                time.sleep(0.005)


def handle_output_df(
    outputs: List[str], model_name: str, visualize_chunk: bool = False
) -> None:
    """
    Handle the output dataframe

    Parameters:
    ----------
    outputs: List[str]
        The list of outputs
    model_name: str
        The model name
    visualize_chunk: bool
        Whether to visualize the chunk
    """
    result_dir = PACKAGE_ROOT_PATH / "results" / model_name
    result_dir.mkdir(exist_ok=True)
    file_path = result_dir / f"result_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    final_df = pd.DataFrame()
    final_str = ""
    for output in outputs:
        try:
            output = eval(output)
            df = pd.DataFrame(output)
            final_df = pd.concat([final_df, df], axis=0)

        except Exception as e:
            try:
                df = handle_corrupted_data(output)
                final_df = pd.concat([final_df, df], axis=0)
            except Exception as e:
                print(e)
                output = str(output)
                final_str += output + "\n"

    if not final_df.empty:
        final_df.reset_index(drop=True, inplace=True)
        file_path = file_path.with_suffix(".xlsx")
        final_df.to_excel(file_path, index=False)
        if not visualize_chunk:
            st.dataframe(final_df, height=600, use_container_width=True)

    if final_str:
        file_path = file_path.with_suffix(".txt")
        with open(file_path, "w") as f:
            f.write(final_str)
        if not visualize_chunk:
            st.write(final_str, unsafe_allow_html=True)


def handle_corrupted_data(output: str) -> pd.DataFrame:
    """
    Handle the corrupted data

    Parameters:
    ----------
    output: str
        The output to handle

    Returns:
    -------
    pd.DataFrame
        The dataframe
    """
    # Extract the two lists
    informal_start = output.find("[")
    informal_end = output.find("]") + 1
    informal_list = output[informal_start:informal_end]

    formal_start = output.find("[", informal_end)
    formal_end = output.rfind("]") + 1
    formal_list = output[formal_start:formal_end]

    # Convert string representation of lists to actual lists
    informal_names = ast.literal_eval(informal_list)
    formal_names = ast.literal_eval(formal_list)

    # Create a DataFrame
    df = pd.DataFrame({"informal_names": informal_names, "formal_names": formal_names})

    return df
