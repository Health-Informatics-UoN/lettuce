from evaluation.pipelines import LLMPipeline, RAGPipeline
from sentence_transformers import SentenceTransformer
from jinja2 import Environment
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever

from options.pipeline_options import LLMModel

llm = LLMModel.LLAMA_3_1_8B
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
store = QdrantDocumentStore(
    path="concept_embeddings.qdrant", embedding_dim=384, recreate_index=False
)
retriever = QdrantEmbeddingRetriever(store)
template_env = Environment()
llm_prompt_template = template_env.from_string(
    """You are an assistant that suggests formal RxNorm names for a medication. You will be given the name of a medication, along with some possibly related RxNorm terms. If you do not think these terms are related, ignore them when making your suggestion.

Respond only with the formal name of the medication, without any extra explanation.

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

Informal name: {{informal_name}}
Response:
"""
)

rag_prompt_template = template_env.from_string(
    """You are an assistant that suggests formal RxNorm names for a medication. You will be given the name of a medication, along with some possibly related RxNorm terms. If you do not think these terms are related, ignore them when making your suggestion.

Respond only with the formal name of the medication, without any extra explanation.

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

Informal name: {{informal_name}}
Response:"""
)

template_vars = ["informal_name", "vec_results"]

pl = LLMPipeline(
    llm=llm, prompt_template=llm_prompt_template, template_vars=["informal_name"]
)

rag = RAGPipeline(
    llm=llm,
    prompt_template=rag_prompt_template,
    template_vars=template_vars,
    embedding_model=embedding_model,
    retriever=retriever,
)

if __name__ == "__main__":
    print(pl.run(["soap"]))
