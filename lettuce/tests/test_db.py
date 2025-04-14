import os
from dotenv import load_dotenv
from os import environ
import pytest


pytestmark = pytest.mark.skipif(os.getenv('SKIP_DATABASE_TESTS') == 'true', reason="Skipping database tests")


from urllib.parse import quote_plus

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from omop import OMOP_match
from omop.omop_queries import (
    query_ancestors_by_name,
    query_descendants_by_name,
    query_ids_matching_name,
    query_related_by_name,
)
from utils.logging_utils import logger

from unittest.mock import Mock
from sqlalchemy.orm import Session
from components.embeddings import PGVectorQuery
from haystack.dataclasses import Document



# --- Testing main OMOP_match --->
@pytest.fixture
def single_query_result():
    return OMOP_match.run(
        search_term=["Acetaminophen"], logger=logger, vocabulary_id=["RxNorm"]
    )


@pytest.fixture
def three_query_result():
    return OMOP_match.run(
        search_term=["Acetaminophen", "Codeine", "Omeprazole"],
        logger=logger,
        vocabulary_id=["RxNorm"],
    )


def test_single_query_returns_one_result(single_query_result):
    assert len(single_query_result) == 1


def test_single_query_keys(single_query_result):
    assert list(single_query_result[0].keys()) == ["search_term", "CONCEPT"]


def test_single_query_concept_keys(single_query_result):
    concept_keys = list(single_query_result[0]["CONCEPT"][0])
    assert concept_keys == [
        "concept_name",
        "concept_id",
        "vocabulary_id",
        "concept_code",
        "concept_name_similarity_score",
        "CONCEPT_SYNONYM",
        "CONCEPT_ANCESTOR",
        "CONCEPT_RELATIONSHIP",
    ]


def test_three_query_returns_three_results(three_query_result):
    assert len(three_query_result) == 3


# --- Testing utility queries -->
#load_dotenv()


@pytest.fixture
def db_connection():
    DB_HOST = environ["DB_HOST"]
    DB_USER = environ["DB_USER"]
    DB_PASSWORD = quote_plus(environ["DB_PASSWORD"])
    DB_NAME = environ["DB_NAME"]
    DB_PORT = environ["DB_PORT"]

    connection_string = (
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    return create_engine(connection_string)


def test_matching_by_name(db_connection):
    Session = sessionmaker(db_connection)
    session = Session()

    query = query_ids_matching_name("acetaminophen", vocabulary_ids=["RxNorm"])
    results = session.execute(query).fetchall()
    session.close()

    assert len(results) == 1
    # This is the concept_id for RxNorm's acetaminophen entry - change if the underlying vocabulary changes
    assert results[0][0] == 1125315


def test_fetch_related_concepts_by_name(db_connection):
    # I'm sure there's a neater pytest-y way to keep the session active, but honestly it's not worth the bother
    Session = sessionmaker(db_connection)
    session = Session()

    query = query_related_by_name("acetaminophen", vocabulary_ids=["RxNorm"])
    results = session.execute(query).fetchall()
    session.close()

    assert len(results) > 1

    names = [result[0].concept_name for result in results]
    assert "Sinutab" in names


def test_fetch_ancestor_concepts_by_name(db_connection):
    Session = sessionmaker(db_connection)
    session = Session()

    query = query_ancestors_by_name("acetaminophen", vocabulary_ids=["RxNorm"])
    results = session.execute(query).fetchall()
    session.close()

    assert len(results) > 1

    names = [result[0].concept_name for result in results]
    assert (
        "LITTLE REMEDIES NEW BABY ESSENTIALS - acetaminophen, simethicone, zinc oxide kit"
        in names
    )


def test_fetch_ancestor_concepts_by_name_with_separation_bounds(db_connection):
    Session = sessionmaker(db_connection)
    session = Session()

    query = query_ancestors_by_name(
        "acetaminophen",
        vocabulary_ids=["RxNorm"],
        min_separation_bound=1,
        max_separation_bound=1,
    )
    results = session.execute(query).fetchall()
    session.close()

    assert len(results) > 1

    names = [result[0].concept_name for result in results]
    assert "homatropine methylbromide; systemic" in names


def test_fetch_descendant_concepts_by_name(db_connection):
    Session = sessionmaker(db_connection)
    session = Session()

    query = query_descendants_by_name("acetaminophen", vocabulary_ids=["RxNorm"])
    results = session.execute(query).fetchall()
    session.close()

    assert len(results) > 1

    names = [result[0].concept_name for result in results]
    assert (
        "Acetaminophen 0.0934 MG/MG / Ascorbic Acid 0.0333 MG/MG / Pheniramine 0.00333 MG/MG Oral Granules [FERVEX RHUME PARACETAMOL/VITAMINE C] Box of 8"
        in names
    )


def test_fetch_descendant_concepts_by_name_with_separation_bounds(db_connection):
    Session = sessionmaker(db_connection)
    session = Session()

    query = query_descendants_by_name("acetaminophen", vocabulary_ids=["RxNorm"])
    results = session.execute(query).fetchall()
    session.close()

    assert len(results) > 1

    names = [result[0].concept_name for result in results]
    assert "Painaid BRF Oral Product" in names

class TestPGVectorQuery:
    def test_initialization(self):
        """
        Test that the component can be correctly initialized with a database session
        """
        mock_session = Mock(spec=Session)
        query_component = PGVectorQuery(connection=mock_session)
        
        assert query_component._connection == mock_session

    def test_run_with_valid_embedding(self):
        """
        Test the run method with a valid query embedding
        """
        # Prepare mock data
        mock_session = Mock(spec=Session)
        mock_results = [
            {
                "id": "concept1", 
                "content": "Sample Concept 1", 
                "score": 0.1
            },
            {
                "id": "concept2", 
                "content": "Sample Concept 2", 
                "score": 0.2
            }
        ]
        
        # Configure mock session to return the prepared results
        mock_execute = Mock()
        mock_execute.mappings.return_value.all.return_value = mock_results
        mock_session.execute.return_value = mock_execute
        
        # Create the component
        query_component = PGVectorQuery(connection=mock_session)
        
        # Run the method
        query_embedding = [0.1, 0.2, 0.3]  # Example embedding
        result = query_component.run(query_embedding=query_embedding, top_k=2)
        
        # Assertions
        assert len(result['documents']) == 2
        assert all(isinstance(doc, Document) for doc in result['documents'])
        assert result['documents'][0].id == "concept1"
        assert result['documents'][0].content == "Sample Concept 1"
        assert result['documents'][0].score == 0.1

    def test_run_with_empty_results(self):
        """
        Test the run method when no results are found
        """
        mock_session = Mock(spec=Session)
        mock_execute = Mock()
        mock_execute.mappings.return_value.all.return_value = []
        mock_session.execute.return_value = mock_execute
        
        query_component = PGVectorQuery(connection=mock_session)
        query_embedding = [0.1, 0.2, 0.3]
        
        result = query_component.run(query_embedding=query_embedding)
        
        assert result['documents'] == []

    def test_run_with_top_k_parameter(self):
        """
        Test that the top_k parameter limits the number of results
        """
        mock_session = Mock(spec=Session)
        mock_results = [
            {"id": f"concept{i}", "content": f"Sample Concept {i}", "score": 0.1 * i} 
            for i in range(1, 6)
        ]
        
        mock_execute = Mock()
        mock_execute.mappings.return_value.all.return_value = mock_results
        mock_session.execute.return_value = mock_execute
        
        query_component = PGVectorQuery(connection=mock_session)
        query_embedding = [0.1, 0.2, 0.3]
        
        # Test with different top_k values
        for k in [1, 3, 5]:
            result = query_component.run(query_embedding=query_embedding, top_k=k)
            assert len(result['documents']) == k

    def test_invalid_embedding_type(self):
        """
        Test handling of invalid embedding types
        """
        mock_session = Mock(spec=Session)
        query_component = PGVectorQuery(connection=mock_session)
        
        with pytest.raises(TypeError):
            query_component.run(query_embedding="not a list")
        
        with pytest.raises(TypeError):
            query_component.run(query_embedding=[1, "invalid", 3])
