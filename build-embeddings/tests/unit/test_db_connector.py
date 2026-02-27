import pytest
from unittest.mock import Mock

from embedding_utils.db_utils import PGConnector


@pytest.fixture
def connector() -> PGConnector:
    return PGConnector(
        db_user="user",
        db_password="password",
        db_host="localhost",
        db_port=5432,
        db_name="name",
        db_schema="schema",
        logger=Mock(),
    )


def test_connector_properties(connector):
    assert connector.db_schema == "schema"
    assert connector.embeddings_table_name == "embeddings"
