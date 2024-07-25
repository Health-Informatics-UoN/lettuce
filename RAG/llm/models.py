def get_model(model, model_name, temperature=0.7):
    if model.lower() == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name, temperature=temperature)


if __name__ == "__main__":
    model = get_model("openai", "gpt-3.5-turbo", temperature=0.0)
    print(model)
