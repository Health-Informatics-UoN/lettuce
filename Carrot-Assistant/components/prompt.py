from haystack.components.builders import PromptBuilder


class Prompts:
    """
    This class is used to generate prompts for the models.
    """

    def __init__(
        self,
        model_name: str,
        prompt_type: str | None = "simple",
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

    def get_prompt(self) -> PromptBuilder | None:
        """
        Get the prompt based on the prompt_type supplied to the object.

        Returns
        -------
        PromptBuilder
            The prompt for the model

            - If the _prompt_type of the object is "simple", returns a simple prompt for few-shot learning of formal drug names.
        """
        if self._prompt_type == "simple":
            return self._simple_prompt()

    def _simple_prompt(self) -> PromptBuilder:
        """
        Get a simple prompt

        Returns
        -------
        PromptBuilder
            The simple prompt
        """
        prompt_template = """
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
"""

        return PromptBuilder(template=prompt_template)
