"""
Template definitions for LLM prompts used in medical term standardization.

This module contains Jinja2 template strings that are used to generate prompts for
different LLM inference scenarios. Templates support domain-specific customization
and variable substitution for flexible prompt generation.

Templates
---------
simple : str
    A few-shot learning template that provides examples of medication name standardization
    without external context. Uses conditional domain rendering.
    
top_n_RAG : str  
    A retrieval-augmented generation template that includes potentially related OMOP
    concept names from vector similarity search. Allows the model to leverage external
    knowledge while making standardisation decisions.

Template Variables
------------------
All templates support the following Jinja2 variables:
- informal_name : str
    The source term to be standardized
- domain : list[str], optional
    List of domain types (e.g., ["drug", "condition"]) for context
- vec_results : str (top_n_RAG only)
    String containing related terms from vector similarity search

Notes
-----
Templates use Jinja2 syntax for conditional rendering and iteration, particularly
for handling domain specification in a grammatically correct manner.
"""

templates = {
        "simple": """You are an assistant that suggests formal OMOP concept names for a source term. Respond only with the formal name of that {% if domain %}
    {%- if domain|length == 1 -%}
    {{ domain[0] }}
    {%- else -%}
        {%- for d in domain -%}
         {% if not loop.last -%}
            {{d}}, 
         {% else -%}
            or {{d}}
         {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
    {%- else -%}
    source term
{%- endif -%}, without any extra explanation.

Examples:

Informal name: Tylenol
Response: Acetaminophen

Informal name: Advil
Response: Ibuprofen

Informal name: Motrin
Response: Ibuprofen

Informal name: Aleve
Response: Naproxen

Task:

Informal name: {{informal_name}}""",
            "top_n_RAG": """You are an assistant that suggests formal OMOP concept names for source terms. You will be given the name of a {% if domain %}
    {%- if domain|length == 1 -%}
    {{ domain[0] }}
    {%- else -%}
        {%- for d in domain -%}
         {% if not loop.last -%}
            {{d}}, {% else -%}
            or {{d}}
         {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
    {%- else -%}
    source term
{%- endif -%}, along with some possibly related OMOP concept names. If you do not think these terms are related, ignore them when making your suggestion.

Respond only with the formal name of the source term, without any extra explanation.

Examples:

Informal name: Tylenol
Response: Acetaminophen

Informal name: Advil
Response: Ibuprofen

Informal name: Motrin
Response: Ibuprofen

Informal name: Aleve
Response: Naproxen

Possible related terms:
{{ vec_results }}
Task:

Informal name: {{informal_name}}""",
        }
