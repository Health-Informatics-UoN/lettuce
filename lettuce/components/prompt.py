from haystack.components.builders import PromptBuilder
from options.pipeline_options import LLMModel
from components.prompt_templates import templates


class Prompts:
    """
    This class is used to generate prompts for the models.
    """

    def __init__(self, model: LLMModel, prompt_type: str = "simple") -> None:
        """
        Initializes the Prompts class

        Parameters
        ----------
        model_name: LLMModel
            The name of the model
        prompt_type: str|None
            The type of prompt to generate
        """
        self._model_name = model.value
        self._prompt_type = prompt_type
        self._eot_token = model.get_eot_token()
        # I hate how the triple-quoted strings look, but if you indent them they preserve the indentation. You can use textwrap.dedent to solve it, but that's not pleasing either.
        # modify this so it only adds the EOT token for llama 3.1
        self._prompt_templates = templates

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
            template = self._prompt_templates[self._prompt_type]
            template += self._eot_token + "\nResponse:"
        except KeyError:
            print(f"No prompt named {self._prompt_type}")
        return PromptBuilder(template)
