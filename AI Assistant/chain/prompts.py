from langchain.prompts import PromptTemplate


class Prompts:
    """
    This class is used to generate prompts for the models.
    """

    def __init__(
        self,
        prompt_type: str | None = None,
        use_memory: bool = False,
        hub: str | None = None,
        model_name: str | None = None,
    ) -> PromptTemplate:
        """
        Initialise the class

        Parameters:
        ----------
        prompt_type: str
            The type of prompt to generate
        use_memory: bool
            Whether to use memory in the prompt
        hub: str
            The hub to use
        model_name: str
            The model name to use

        Returns:
        -------
        PromptTemplate
            The prompt template
        """
        self.prompt_type = prompt_type
        self.use_memory = use_memory
        self.hub = hub
        self.model_name = model_name

    def get_prompt(self) -> PromptTemplate:
        """
        Get the prompt

        Returns:
        -------
        PromptTemplate
            The prompt template
        """
        if self.prompt_type == "simple":
            return self._simple_prompt()
        else:
            if self.prompt_type == "conversion":
                if "llamacpp" in self.hub.lower():
                    return self._medicine_conversion_Llama(use_memory=self.use_memory)
                else:
                    return self._medicine_conversion(use_memory=self.use_memory)

    def _simple_prompt(self) -> PromptTemplate:
        """
        Generate a simple prompt

        Returns:
        -------
        PromptTemplate
            The prompt template
        """
        template = """[INST]What are the formal names of medications:{informal_names}-{informal_names_length}?[/INST]"""
        return PromptTemplate.from_template(template)

    def _medicine_conversion(self, use_memory: bool = False) -> PromptTemplate:
        """
        Generate a medicine conversion prompt

        Parameters:
        ----------
        use_memory: bool
            Whether to use memory in the prompt

        Returns:
        -------
        PromptTemplate
            The prompt template
        """
        template = """\
        You are an AI assistant for the pharmaceutical department at the University of Nottingham. \
        Your task is to process a dataframe containing informal names of medications and convert them into \
        the respective formal drug names, utilizing your extensive knowledge base. \
        You will receive the dataframe as input called "informal_names" which contains a list of informal names of medications. \
        When producing the output, you must follow these guidelines: \
        - The produced output should be a dictionary. \
        - The dictionary should have two keys: "informal_names" and "formal_names" and the values should be lists of the same length. \
        - The produced "informal_names" should be same as the user input. Do not change it. \
        - The produced "formal_names" should be complete and not partial. \
        - The length of the input informal names is {informal_names_length}. The produced output length should be equal to the length of the input informal names for both keys. It is a mandatory requirement. \
        - The produced output should be in a format to be used to import into a pandas dataframe. \
        - Don't produce any other output or sentence rather than the dataframe. \
        - If you don't know the formal name of a medicine, don't try to make up a name or repeat the informal name. \
        Here is the examples of the format of the user input and the expected output you should produce: \
        Example: \
        user_input: \
        [Document(page_content='Ppaliperidone (3-month)'), Document(page_content='Latanoprost 0.005% (Both Eye)'), Document(page_content='Euthyrox (Sun)'), Document(page_content='Dapagliflozin'), Document(page_content='Humalog 32/22'), Document(page_content='Telmisartan/Amlodipine'), Document(page_content='Ashwagandha')] \
        expected_output: \
        informal_names=["Ppaliperidone (3-month)", "Latanoprost 0.005% (Both Eye)", "Euthyrox (Sun)", "Dapagliflozin", "Humalog 32/22", "Telmisartan/Amlodipine", "Ashwagandha"], formal_names=["Paliperidone", "Latanoprost", "Levothyroxine", "Dapagliflozin", "Insulin lispro", "Telmisartan/Amlodipine", "Withania somnifera"]

        informal_names:
        {informal_names}

        AI Assistant Output:     
        """
        if use_memory:
            template = template.replace(
                "AI Assistant Output:",
                "Chat History:\n{chat_history}\n\nAI Assistant Output:",
            )
        return PromptTemplate.from_template(template)

    def _medicine_conversion_Llama(self, use_memory: bool = False) -> PromptTemplate:
        """
        Edit the medicine conversion prompt for Llama models

        Parameters:
        ----------
        use_memory: bool
            Whether to use memory in the prompt

        Returns:
        -------
        PromptTemplate
            The prompt template
        """
        prompt = self._medicine_conversion(use_memory=use_memory)
        prompt.template = prompt.template.replace(
            "You are an AI assistant",
            "[INST] <<SYS>>\nYou are an AI assistant",
        )
        prompt.template = prompt.template.replace(
            "informal_names:",
            "<</SYS>>\ninformal_names:",
        )
        prompt.template = prompt.template.replace(
            "AI Assistant Output:",
            "AI Assistant Output: [/INST]",
        )

        return prompt
