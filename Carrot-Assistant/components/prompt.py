from haystack.components.builders import PromptBuilder


class Prompts:
    """
    This class is used to generate prompts for the models.
    """

    def __init__(
        self,
        model_name: str,
        prompt_type: str = "simple",
    ) -> None:
        """
        Initializes the Prompts class

        Parameters
        ----------
        model_name: str
            The name of the model
        prompt_type: str|None
            The type of prompt to generate
        """
        self._model_name = model_name
        self._prompt_type = prompt_type
        # I hate how the triple-quoted strings look, but if you indent them they preserve the indentation. You can use textwrap.dedent to solve it, but that's not pleasing either.
        # modify this so it only adds the EOT token for llama 3.1
        self._prompt_templates = {
                "simple": """
You will be given the informal name of a medication. Respond only with the formal name of that medication, without any extra explanation.

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

Informal name: {{informal_name}}<|eot_id|>
Response:
""",
                "top_n_RAG": """
You are an assistant that suggests formal RxNorm names for a medication. You will be given the name of a medication, along with some possibly related RxNorm terms. If you do not think these terms are related, ignore them when making your suggestion.

Respond only with the formal name of the medication, without any extra explanation.

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
Informal name: {{informal_name}}<|eot_id|>
Response:
""",
                }

    def get_prompt(self) -> PromptBuilder:
        """
        Get the prompt based on the prompt_type supplied to the object.

        Returns
        -------
        PromptBuilder
            The prompt for the model

            - If the _prompt_type of the object is "simple", returns a simple prompt for few-shot learning of formal drug names.
        """
        try:
            return PromptBuilder(self._prompt_templates[self._prompt_type])
        except KeyError:
            print(f"No prompt named {self._prompt_type}")
        
