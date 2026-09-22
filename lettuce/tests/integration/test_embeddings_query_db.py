import os
import pytest

from components.embeddings import Embeddings, PGVectorQuery
from options.base_options import BaseOptions

pytestmark = pytest.mark.skipif(os.getenv('SKIP_DATABASE_TESTS') == 'true', reason="Skipping database tests")


@pytest.fixture
def embedding_handler() -> Embeddings:
    settings = BaseOptions()
    return Embeddings(model_name=settings.embedding_model)

@pytest.fixture
def retriever(embedding_handler: Embeddings) -> PGVectorQuery:
    return embedding_handler.get_retriever()

@pytest.fixture
def example_embedding(embedding_handler) -> list[float]:
    embedder = embedding_handler.get_embedder()
    return embedder.run("Wobbly legs")["embedding"]

def test_top_k(retriever: PGVectorQuery, example_embedding):
    for k in [1,3,5]:
        results = retriever.run(
                example_embedding,
                domain_id=["Observation"],
                top_k = k)
        assert len(results["documents"]) == k

def test_domain_id(retriever: PGVectorQuery, example_embedding):
    for domain in [["Observation"], ["Condition"]]:
        results = retriever.run(
                example_embedding,
                domain_id = domain,
                describe_concept=True
                )
        retrieved_domains = [res["Concept"].domain_id for res in results]
        print(retrieved_domains)
        assert all(d in domain for d in retrieved_domains)
