import os 
import pandas as pd 
import pytest 
from haystack_integrations.components.embedders.fastembed.fastembed_text_embedder import FastembedTextEmbedder

from components.embeddings import (
    PGVectorQuery, 
    Embeddings, 
    EmbeddingModelName
)


TEST_EMBED_MODEL_NAME = EmbeddingModelName("BGESMALL") 
TEST_EMBED_VOCAB = "RxNorm"
TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_TO_TEST_DATA = os.path.join(TEST_DIR, "test_data", "acetaminophen_embedding_bgesmall.csv")
ACETAMINOPHEN_BGESMALL_EMBED = pd.read_csv(PATH_TO_TEST_DATA, header=None)[0].tolist()


@pytest.fixture
def single_vector_query_result():
    return PGVectorQuery(
        embed_vocab=[TEST_EMBED_VOCAB], 
    ).run(query_embedding=ACETAMINOPHEN_BGESMALL_EMBED) 


@pytest.fixture 
def embeddings_instance(): 
    return Embeddings(
        model_name=TEST_EMBED_MODEL_NAME, 
        embed_vocab=TEST_EMBED_VOCAB, 
        standard_concept=False, 
        top_k=5 
    )


class TestPGVectorQuery(): 
    def test_run(self, single_vector_query_result): 
        best_match = single_vector_query_result["documents"][0]
        assert best_match.content == "acetaminophen"
        assert best_match.score < 1e-6


class TestEmbeddings(): 
    def test_get_embedder(self, embeddings_instance): 
        vector_embedder = embeddings_instance.get_embedder() 

        assert isinstance(vector_embedder, FastembedTextEmbedder)

        text = "Hello"
        embedding = vector_embedder.run(text)["embedding"]
        
        assert embedding is not None
        assert len(embedding) > 0
        assert isinstance(embedding, list)
        assert all(isinstance(x, float) for x in embedding)
        assert len(embedding) ==  int(os.getenv("DB_VECSIZE"))
        assert not all(x == 0.0 for x in embedding)
        