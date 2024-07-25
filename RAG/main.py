import os
import sys
from dotenv import load_dotenv
from constant import PACKAGE_ROOT_PATH
from llm.chains import Chains
from llm.agents import Agents
from llm.prompts import Prompts
from options.rag_option import RAGOptions
from preprocessing.vectors import csv_folder_retriever
from langchain_community.vectorstores import FAISS, Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.callbacks.tracers import ConsoleCallbackHandler


# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


load_dotenv()


def run() -> None:
    opt = RAGOptions().parse()
    problem_type = opt.problem_type
    main_data_folder = opt.main_data_folder
    use_multithreading = opt.use_multithreading

    prompt_template = Prompts().get_prompt("omop_structure")
    question = input("Enter your question: ")
    
    # """
    # Find the concept name and code of the tradename of the drug 'penicillin V 125' Capsule.
    # """

    prompt = prompt_template.format(question=question)

    if problem_type == "csv_folder_retriever":
        main_retriever = csv_folder_retriever(opt, main_data_folder, use_multithreading)
        chain = Chains(opt, main_retriever).get_chain()
        res = chain.invoke({"question": question})

    elif problem_type == "csv_agent":
        csv_agent = Agents(opt)
        csv_agent.get_csv_agent()
        res = csv_agent.invoke(prompt)

    print(res["output"])


if __name__ == "__main__":
    run()

# TODO
# Add logger
