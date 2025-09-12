templates = {
        "simple": """You are an assistant that suggests formal OMOP concept names for a {{domain}}. Respond only with the formal name of that {{domain}}, without any extra explanation.

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
            "top_n_RAG": """You are an assistant that suggests formal OMOP concept names for a {{domain}}. You will be given the name of a {{domain}}, along with some possibly related OMOP concept names. If you do not think these terms are related, ignore them when making your suggestion.

Respond only with the formal name of the {{domain}}, without any extra explanation.

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
{% for result in vec_results %}
    {{result.content}}
{% endfor %}

Task:

Informal name: {{informal_name}}""",

        }
