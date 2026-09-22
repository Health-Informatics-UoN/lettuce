from pydantic import BaseModel
from jinja2 import Template


class RenderedConcept(BaseModel):
    """Dataclass for a Concept with the rendered string, ready for encoding"""
    concept_id: int
    concept_name: str
    concept_string: str


class Concept(BaseModel):
    """
    Dataclass for concepts as read from some source
    Attributes taken from the OMOP concept table

    Attributes
    ----------
    concept_id: int
    concept_name: str
    domain:str
    vocabulary:str
    concept_class: str
    """
    concept_id: int
    concept_name: str
    domain: str
    vocabulary: str
    concept_class: str

    def render_concept_as_template(self, template: Template) -> RenderedConcept:
        """
        Takea the concept's attributes and a jinja2 Template and uses them to render a string for encoding

        Parameters
        ----------
        template: Template
            A jinja2 Template that can be used to render a string for the concept

        Returns
        -------
        RenderedConcept
        """
        return RenderedConcept(
            concept_id=self.concept_id,
            concept_name=self.concept_name,
            concept_string=template.render(self.model_dump()),
        )
