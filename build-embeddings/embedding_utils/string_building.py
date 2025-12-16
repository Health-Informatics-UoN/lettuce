from pydantic import BaseModel
from jinja2 import Template

class Concept(BaseModel):
    concept_id: int
    concept_name: str
    domain: str
    vocabulary: str
    concept_class: str

    def render_concept_as_template(self, template: Template) -> tuple[int, str, str]:
        return (self.concept_id, self.concept_name, template.render(self.model_dump()))
