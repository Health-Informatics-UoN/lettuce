from haystack.components.builders import PromptBuilder
from components.prompt_templates import templates


class Prompts:
    """
    This class is used to generate prompts for the models.
    
    The Prompts class manages template selection and prompt building for different LLM models,
    automatically handling model-specific formatting requirements such as end-of-turn tokens.
    It supports multiple prompt types including simple few-shot learning and retrieval-augmented
    generation approaches.
    """

    def __init__(self, prompt_type: str = "simple") -> None:
        """
        Initializes the Prompts class.

        Parameters
        ----------
        model : LLMModel
            The LLM model enum containing model name and configuration details
        prompt_type : str, optional
            The type of prompt to generate. Defaults to "simple".
            Available options:
            - "simple": Few-shot learning prompt without external data
            - "top_n_RAG": Retrieval-augmented generation prompt with related terms
        """
        self._prompt_type = prompt_type
        self._prompt_templates = templates

    def get_prompt(self) -> PromptBuilder:
        """
        Get the prompt based on the prompt_type supplied to the object.

        Retrieves the appropriate template based on the prompt type, appends the model's
        end-of-turn token, and returns a configured PromptBuilder instance.

        Returns
        -------
        PromptBuilder
            A configured Haystack PromptBuilder object with the selected template.
            The template includes model-specific formatting and is ready for rendering
            with the required variables (informal_name, domain, vec_results as applicable).

        Raises
        ------
        KeyError
            If the specified prompt_type is not found in the available templates.
            Prints an error message and continues execution.

        Notes
        -----
        The method automatically appends the model's end-of-turn token and a "Response:"
        prompt to guide the model's output formatting.
        """
        try:
            template = self._prompt_templates[self._prompt_type]
        except KeyError:
            print(f"No prompt named {self._prompt_type}")
        return PromptBuilder(template)
