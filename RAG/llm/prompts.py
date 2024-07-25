from langchain.prompts import PromptTemplate


class Prompts:
    def __init__(self):
        pass

    def get_prompt(self, prompt_type):
        if prompt_type.lower() == "simple_retrieval":
            return self._simple_retrieval_prompt()
        elif prompt_type.lower() == "omop_structure":
            return self._OMOP_structure_prompt()

    def _simple_retrieval_prompt(self):
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        return PromptTemplate.from_template(template)

    def _OMOP_structure_prompt(self):
        template = """
        You are given the dataframes of OMOP which are related to each other based on common ids. There are 3 main dataframes: CONCEPT, CONCEPT_ANCESTOR, CONCEPT_RELATIONSHIP.
        CONCEPT DataFrame:
        'domain_id' is a column in the concept table. It represents the domain to which the concept belongs, e.g. 'Drug', 'Condition', 'Procedure', etc.
        'standard_concept' is a column in the concept table. It represents whether the concept is standard or not. If it is standard, it is specified as 'S'.
        'concept_name' is a column in the concept table. It represents the name of the concept. When searching for a concept, the name might be a part of the name, so search should be done using "contain" operator. Also it should not be case sensitive.
        'concept_id' is a column in the concept table. It represents the unique identifier for the concept.
        'concept_code' is a column in the concept table. It represents the code that is used to identify the concept.
        'concept_class_id' is a column in the concept table. It represents the class of the concept, e.g. 'Clinical Finding', 'Drug', 'Procedure', 'Clinical Drug', etc.

        CONCEPT_ANCESTOR DataFrame:
        'ancestor_concept_id' is a column in the concept_ancestor table. It represents the unique identifier for the ancestor concept.
        'descendant_concept_id' is a column in the concept_ancestor table. It represents the unique identifier for the descendant concept.

        CONCEPT_RELATIONSHIP DataFrame:
        'concept_id_1' is a column in the concept_relationship table. It represents the unique identifier for the first concept.
        'concept_id_2' is a column in the concept_relationship table. It represents the unique identifier for the second concept.
        'relationship_id' is a column in the concept_relationship table. It represents the relationship between the two concepts, e.g. 'Maps to', 'Is a', 'Has ingredient', 'Has tradename', etc.

        Answer the question based on the previous instructions and the above context. Be sure you don't have unexpected indent in the code.
        Question: {question}
        """
        return PromptTemplate.from_template(template)


if __name__ == "__main__":
    prompt = Prompts().get_prompt("simple_retrieval")
    print(prompt)
