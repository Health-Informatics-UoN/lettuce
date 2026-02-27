from sentence_transformers import SentenceTransformer
from jinja2 import Template

from embedding_utils.protocols import ConceptEmbedder, EmbeddedConcept
from embedding_utils.string_building import Concept


class BatchEmbedder(ConceptEmbedder):
    def __init__(
        self,
        embedding_model: SentenceTransformer,
        template: Template,
    ) -> None:
        super().__init__()
        self._embedding_model = embedding_model
        self._template = template

    @property
    def dimension(self):
        return self._embedding_model.get_sentence_embedding_dimension()

    def embed_concepts(self, concepts: list[Concept]) -> list[EmbeddedConcept]:
        concept_strings = [
            c.render_concept_as_template(self._template) for c in concepts
        ]
        concept_embeddings = self._embedding_model.encode(
            [c.concept_string for c in concept_strings],
            convert_to_tensor=False,
            show_progress_bar=True,
        ).tolist()
        return list(
            [
                EmbeddedConcept(
                    concept_id=concept.concept_id,
                    concept_name=concept.concept_name,
                    embedding=embedding,
                )
                for concept, embedding in zip(concept_strings, concept_embeddings)
            ]
        )
