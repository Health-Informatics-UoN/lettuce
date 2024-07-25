from langchain.memory import ConversationBufferMemory


def get_memory(
    prev_memory: ConversationBufferMemory | None = None,
    input_key: str = "user_question",
) -> ConversationBufferMemory:
    """
    Get the memory

    Parameters:
    ----------
    prev_memory: ConversationBufferMemory
        The previous memory
    input_key: str
        The input key to use

    Returns:
    -------
    ConversationBufferMemory
        The memory
    """
    memory_key = "chat_history"
    ai_prefix = "AI Assistant"
    human_prefix = "User"
    memory = ConversationBufferMemory(
        memory_key=memory_key,
        return_messages=True,
        human_prefix=human_prefix,
        ai_prefix=ai_prefix,
        input_key=input_key,
    )
    if prev_memory:
        memory.input_key = "user_question"
        inputs = prev_memory[0].content
        outputs = prev_memory[1].content
        memory.save_context({input_key: inputs, "outputs": outputs})

    return memory
