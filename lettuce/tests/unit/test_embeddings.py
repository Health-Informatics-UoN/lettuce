import pytest
from components.embeddings import PGVectorQuery


class TestPGVectorQuery:
    def test_invalid_embedding_type(self):
        """
        Test handling of invalid embedding types
        """
        query_component = PGVectorQuery()

        with pytest.raises(TypeError):
            query_component.run(query_embedding="not a list")

        with pytest.raises(TypeError):
            query_component.run(query_embedding=[1, "invalid", 3])
