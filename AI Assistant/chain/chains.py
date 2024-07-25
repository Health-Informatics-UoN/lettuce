from typing import Dict, Union

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

from chain.memory import get_memory
from chain.models import get_model
from chain.prompts import Prompts


class Chains:
    """
    This class is used to generate the LLM chain.
    """

    def __init__(
        self,
        chain_type: str | None = None,
        llm_model: Dict | None = None,
        temperature: float = 0.7,
        prev_memory: ConversationBufferMemory | None = None,
        use_memory: bool = False,
        memory_input_key: str = "user_question",
        use_simple_prompt: bool = False,
    ):
        """
        Initialise the class

        Parameters:
        ----------
        chain_type: str
            The type of chain to generate
        llm_model: ChatOpenAI|LlamaCpp|GPT4All
            The model to use
        temperature: float
            The temperature to use
        prev_memory: ConversationBufferMemory
            The previous memory
        use_memory: bool
            Whether to use memory
        memory_input_key: str
            The memory input key
        use_simple_prompt: bool
            Whether to use a simple prompt
        """
        self.chain_type = chain_type.lower()
        self.hub = llm_model["hub"]
        self.model_name = llm_model["model_name"]
        self.temperature = temperature
        self.prev_memory = prev_memory
        self.use_memory = use_memory
        self.memory_input_key = memory_input_key
        if use_simple_prompt:
            self.prompt_type = "simple"
        else:
            self.prompt_type = self.chain_type

    def get_chain(self) -> LLMChain:
        """
        Get the chain

        Returns:
        -------
        LLMChain
            The LLM chain
        """
        prompt = Prompts(
            prompt_type=self.prompt_type,
            use_memory=self.use_memory,
            hub=self.hub,
            model_name=self.model_name,
        ).get_prompt()
        memory = None
        if self.use_memory:
            memory = get_memory(
                prev_memory=self.prev_memory, input_key=self.memory_input_key
            )
        return self._conversation_chain(memory=memory, prompt=prompt)

    def _conversation_chain(
        self, memory: ConversationBufferMemory, prompt: Prompts
    ) -> LLMChain:
        """
        Generate the conversation chain

        Parameters:
        ----------
        memory: ConversationBufferMemory
            The memory
        prompt: Prompts
            The prompt

        Returns:
        -------
        LLMChain
            The LLM chain
        """
        llm = get_model(
            hub=self.hub, model_name=self.model_name, temperature=self.temperature
        )
        memory = memory
        chain = LLMChain(
            llm=llm,
            prompt=prompt,
            memory=memory,
            verbose=True,
        )
        return chain
