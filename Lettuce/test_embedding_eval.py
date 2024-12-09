from sentence_transformers import SentenceTransformer
from evaluation.eval_data_loaders import SingleInputSimpleCSV
from evaluation.eval_tests import EmbeddingComparisonTest
from evaluation.evaltypes import EvaluationFramework
from evaluation.pipelines import EmbeddingsPipeline
from evaluation.metrics import DotVectorSimilarityMetric, CosineVectorSimilarityMetric

dataset = SingleInputSimpleCSV("evaluation/datasets/example.csv")


def make_embedding_tests(model_name):
    model = SentenceTransformer(model_name)
    pipeline = EmbeddingsPipeline(model)
    metric_1 = DotVectorSimilarityMetric(model)
    metric_2 = CosineVectorSimilarityMetric(model)

    return EmbeddingComparisonTest(model_name, pipeline, [metric_1, metric_2])


tests = [
    make_embedding_tests(name)
    for name in ["neuml/pubmedbert-base-embeddings", "BAAI/bge-small-en-v1.5"]
]

if __name__ == "__main__":
    framework = EvaluationFramework(
        "Embeddings Test",
        tests,
        dataset,
        "A small test of the cosine similarity metric for two embeddings models",
        "embedding_results.json",
    )

    framework.run_evaluations()
