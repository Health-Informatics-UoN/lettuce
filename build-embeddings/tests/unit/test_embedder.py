import pytest
from sentence_transformers import SentenceTransformer
from jinja2 import Template, Environment

from embedding_utils.protocols import EmbeddedConcept
from embedding_utils.string_building import Concept
from embedding_utils.embedder import BatchEmbedder


@pytest.fixture
def concepts() -> list[Concept]:
    return [
        Concept(
            concept_id=4323688,
            concept_name="Cough at rest",
            domain="Condition",
            vocabulary="SNOMED",
            concept_class="Clinical Finding",
        ),
        Concept(
            concept_id=4280520,
            concept_name="Pulse taking",
            domain="Measurement",
            vocabulary="SNOMED",
            concept_class="Procedure",
        ),
    ]


@pytest.fixture
def template() -> Template:
    template_env = Environment()
    return template_env.from_string("{{concept_name}}, a {{concept_class}} {{domain}}")


@pytest.fixture
def embedder(template) -> BatchEmbedder:
    return BatchEmbedder(SentenceTransformer("BAAI/bge-small-en-v1.5"), template)


def test_embedder_dimension(embedder):
    assert embedder.dimension == 384


def test_batch_embedder(concepts, embedder):
    embeddings = embedder.embed_concepts(concepts)

    assert len(embeddings) == 2
    example_embedding = embeddings[0]
    assert isinstance(example_embedding, EmbeddedConcept)
    assert hasattr(example_embedding, "concept_id")
    assert hasattr(example_embedding, "concept_name")
    assert hasattr(example_embedding, "embedding")
    assert isinstance(example_embedding.concept_id, int)
    assert isinstance(example_embedding.concept_name, str)
    assert isinstance(example_embedding.embedding, list)
    for val in example_embedding.embedding:
        assert isinstance(val, float)
