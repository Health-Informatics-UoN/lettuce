import csv
import os

import pytest 
from haystack_integrations.components.embedders.fastembed.fastembed_text_embedder import FastembedTextEmbedder

from components.embeddings import PGVectorQuery, Embeddings, EmbeddingModelName
from options.base_options import BaseOptions 


settings = BaseOptions()


TEST_EMBED_MODEL_NAME = EmbeddingModelName("BGESMALL") 
TEST_EMBED_VOCAB = "RxNorm"
TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_TO_TEST_DATA = os.path.join(TEST_DIR, "test_data", "acetaminophen_embedding_bgesmall.csv")

with open(PATH_TO_TEST_DATA, newline="") as f:
    ACETAMINOPHEN_BGESMALL_EMBED = [float(row[0]) for row in csv.reader(f) if row]


@pytest.fixture
def single_vector_query_result():
    return PGVectorQuery().run(query_embedding=ACETAMINOPHEN_BGESMALL_EMBED) 


@pytest.fixture 
def embeddings_instance(): 
    return Embeddings(TEST_EMBED_MODEL_NAME)

@pytest.fixture
def retriever(embeddings_instance):
    return embeddings_instance.get_retriever()

class TestEmbeddings: 
    def test_get_embedder(self, embeddings_instance): 
        vector_embedder = embeddings_instance.get_embedder() 

        assert isinstance(vector_embedder, FastembedTextEmbedder)

        text = "Hello"
        embedding = vector_embedder.run(text)["embedding"]
        
        assert embedding is not None
        assert len(embedding) > 0
        assert isinstance(embedding, list)
        assert all(isinstance(x, float) for x in embedding)
        assert len(embedding) ==  settings.db_vecsize
        assert not all(x == 0.0 for x in embedding)

class TestPGVectorQuery: 
    def test_run(self, single_vector_query_result): 
        best_match = single_vector_query_result["documents"][0]
        assert best_match.content.lower() == "acetaminophen"
        assert best_match.score < 1e-6

    def test_top_k(self, retriever):
        for k in [1,3,5]:
            results = retriever.run(
                    ACETAMINOPHEN_BGESMALL_EMBED,
                    domain_id=["Observation"],
                    top_k = k)
            assert len(results["documents"]) == k
    
    def test_domain_id(self, retriever):
        for domain in [["Observation"], ["Condition"]]:
            results = retriever.run(
                    ACETAMINOPHEN_BGESMALL_EMBED,
                    domain_id = domain,
                    describe_concept=True
                    )
            retrieved_domains = [res["Concept"].domain_id for res in results]
            print(retrieved_domains)
            assert all(d in domain for d in retrieved_domains)
    
            
    
