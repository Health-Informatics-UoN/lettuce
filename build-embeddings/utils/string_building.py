from pydantic import BaseModel
from jinja2 import Template

class Concept(BaseModel):
    concept_id: int
    concept_name: str
    domain: str
    vocabulary: str
    concept_class: str

def render_concept_as_template(template: Template, concept: Concept) -> tuple[int, str, str]:
    return (concept.concept_id, concept.concept_name, template.render(concept.model_dump()))
