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
