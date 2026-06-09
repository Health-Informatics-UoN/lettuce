# String building

## Classes
### `RenderedConcept`
```python
class RenderedConcept(BaseModel):
    concept_id: int
    concept_name: str
    concept_string: str
```

Dataclass for a Concept with the rendered string, ready for encoding


### `Concept`
```python
class Concept(BaseModel):
    concept_id: int
    concept_name: str
    domain: str
    vocabulary: str
    concept_class: str
```

Dataclass for concepts as read from some source.
Attributes taken from the OMOP concept table

#### Methods
##### `render_concept_as_template`
```python
render_concept_as_template(template: Template) -> RenderedConcept:
```
Takes the concept's attributes and a jinja2 Template and uses them to render a string for encoding

| Parameter | Type | Description |
| --------- | --- | --- |
| `template` | Template | A jinja2 Template that can be used to render a string for the concept |

###### Returns `RenderedConcept`
Concept identifiers and the rendered string
