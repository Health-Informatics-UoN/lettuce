from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_embeddings(embedding_model, embedding_model_name):
    if embedding_model.lower() == "openai":
        embedding = OpenAIEmbeddings(model=embedding_model_name)

    elif embedding_model.lower() == "huggingface":
        embedding = HuggingFaceEmbeddings(
            model_name=embedding_model_name, model_kwargs={"device": device}
        )

    else:
        raise NotImplementedError(f"Embedding model {embedding_model} not supported")
    return embedding


if __name__ == "__main__":
    embedding = get_embeddings("HuggingFace", "hkunlp/instructor-xl")
    print(embedding)
