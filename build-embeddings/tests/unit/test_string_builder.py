import pytest
from jinja2.environment import Environment, Template

from embedding_utils.string_building import Concept, RenderedConcept

@pytest.fixture
def concepts() -> list[Concept]:
    return [Concept(
                concept_id=4323688,
                concept_name="Cough at rest",
                domain="Condition",
                vocabulary="SNOMED",
                concept_class="Clinical Finding",
                ),
            Concept(
                concept_id=4280520,
                concept_name='Pulse taking',
                domain="Measurement",
                vocabulary='SNOMED',
                concept_class="Procedure",
                )
            ]

@pytest.fixture
def template() -> Template:
    template_env = Environment()
    return template_env.from_string("{{concept_name}}, a {{concept_class}} {{domain}}")

def test_strings_render(concepts, template):
    concept_strings = [c.render_concept_as_template(template) for c in concepts]
    first_rendered = RenderedConcept(
            concept_id=4323688,
            concept_name="Cough at rest",
            concept_string="Cough at rest, a Clinical Finding Condition"
            )
    second_rendered = RenderedConcept(
            concept_id=4280520,
            concept_name="Pulse taking",
            concept_string="Pulse taking, a Procedure Measurement"
            )
    assert(concept_strings[0] == first_rendered)
    assert(concept_strings[1] == second_rendered)
