from pydantic import BaseModel
from jinja2 import Template


class RenderedConcept(BaseModel):
    concept_id: int
    concept_name: str
    concept_string: str


class Concept(BaseModel):
    concept_id: int
    concept_name: str
    domain: str
    vocabulary: str
    concept_class: str

    def render_concept_as_template(self, template: Template) -> RenderedConcept:
        return RenderedConcept(
            concept_id=self.concept_id,
            concept_name=self.concept_name,
            concept_string=template.render(self.model_dump()),
        )
