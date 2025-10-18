import os
from options.base_options import BaseOptions
import pytest
from sqlalchemy import create_engine, sql
from sqlalchemy.orm import sessionmaker

from omop.omop_queries import (
    query_ancestors_by_name,
    query_descendants_by_name,
    query_ids_matching_name,
    query_related_by_name,
    query_ancestors_and_descendants_by_id, 
    query_related_by_id, 
    text_search_query 
)
from omop.preprocess import preprocess_search_term 


pytestmark = pytest.mark.skipif(os.getenv('SKIP_DATABASE_TESTS') == 'true', reason="Skipping database tests")

settings = BaseOptions()


@pytest.fixture
def db_connection():
    return create_engine(settings.connection_url())


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

    assert len(results) >= 1

    names = [result[0].concept_name for result in results]
    assert "Sinutab" in names


def test_fetch_ancestor_concepts_by_name(db_connection):
    Session = sessionmaker(db_connection)
    session = Session()

    query = query_ancestors_by_name("acetaminophen", vocabulary_ids=["RxNorm"])
    results = session.execute(query).fetchall()
    session.close()

    assert len(results) >= 1

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
    
    assert len(results) >= 1

    names = [result[0].concept_name for result in results]
    assert "homatropine methylbromide; systemic" in names


def test_fetch_descendant_concepts_by_name(db_connection):
    Session = sessionmaker(db_connection)
    session = Session()

    query = query_descendants_by_name("acetaminophen", vocabulary_ids=["RxNorm"])
    results = session.execute(query).fetchall()
    session.close()

    assert len(results) >= 1

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


def test_query_descendants_and_ancestors_by_id(db_connection): 
    concept_id = 1125315  # Acetaminophen

    Session = sessionmaker(db_connection)
    session = Session()

    query = query_ancestors_and_descendants_by_id(concept_id)
    results = session.execute(query).fetchall()
    session.close()
  
    assert len(results) > 1
    
    # Extract data without pandas
    names = [row[4] for row in results]  # concept_name is at index 4
    relationship_types = {row[0] for row in results}  # relationship_type is at index 0

    assert relationship_types == {"Ancestor", "Descendant"}
    assert "Painaid BRF Oral Product" in names
    assert "homatropine methylbromide; systemic" in names 


def test_query_related_by_id(db_connection):
    concept_id = 1125315  # Acetaminophen

    Session = sessionmaker(db_connection)
    session = Session()

    query = query_related_by_id(concept_id)
    results = session.execute(query).fetchall()
    session.close()

    assert len(results) >= 1

    names = [row[4] for row in results]  # concept_name is at index 4
    assert "Sinutab" in names


def test_full_text_query(db_connection):
    search_term = preprocess_search_term("Nervous System")
    query = text_search_query(
        search_term, 
        vocabulary_id=None, 
        standard_concept=True, 
        concept_synonym=False 
    )
    Session = sessionmaker(db_connection)
    session = Session()
    results = session.execute(query).fetchall()
    session.close()
  
    assert len(results) > 1

    expected_entry = (4134440, 'Visual system disorder', 'SNOMED', '128127008', None)
    assert expected_entry in results 


def test_regression_query_descendants_and_ancestors(db_connection): 
    query = f"""
            (
                SELECT
                    'Ancestor' as relationship_type,
                    ca.ancestor_concept_id AS concept_id,
                    ca.ancestor_concept_id,
                    ca.descendant_concept_id,
                    c.concept_name,
                    c.vocabulary_id,
                    c.concept_code,
                    ca.min_levels_of_separation,
                    ca.max_levels_of_separation
                FROM
                    {settings.db_schema}.concept_ancestor ca
                JOIN
                    {settings.db_schema}.concept c ON ca.ancestor_concept_id = c.concept_id
                WHERE
                    ca.descendant_concept_id = :concept_id AND
                    ca.min_levels_of_separation >= :min_separation_ancestor AND
                    ca.max_levels_of_separation <= :max_separation_ancestor
            )
            UNION
            (
                SELECT
                    'Descendant' as relationship_type,
                    ca.descendant_concept_id AS concept_id,
                    ca.ancestor_concept_id,
                    ca.descendant_concept_id,
                    c.concept_name,
                    c.vocabulary_id,
                    c.concept_code,
                    ca.min_levels_of_separation,
                    ca.max_levels_of_separation
                FROM
                    {settings.db_schema}.concept_ancestor ca
                JOIN
                    {settings.db_schema}.concept c ON ca.descendant_concept_id = c.concept_id
                WHERE
                    ca.ancestor_concept_id = :concept_id AND
                    ca.min_levels_of_separation >= :min_separation_descendant AND
                    ca.max_levels_of_separation <= :max_separation_descendant
            )
        """
    concept_id = 1125315
    min_separation_ancestor = 1
    min_separation_descendant = 1
    max_separation_ancestor = 1
    max_separation_descendant = 1

    params = {
            "concept_id": concept_id,
            "min_separation_ancestor": min_separation_ancestor,
            "max_separation_ancestor": max_separation_ancestor,
            "min_separation_descendant": min_separation_descendant,
            "max_separation_descendant": max_separation_descendant,
    }
    
    # Execute original query
    Session = sessionmaker(db_connection)
    session = Session()
    results_original_raw = session.execute(sql.text(query), params).fetchall()
    
    # Remove duplicates and filter out self-references
    seen = set()
    results_original = []
    for row in results_original_raw:
        row_tuple = tuple(row)
        if row_tuple not in seen and row[1] != concept_id:  # concept_id is at index 1
            seen.add(row_tuple)
            results_original.append(row_tuple)
    
    # New query using SQLAlchemy
    query_new = query_ancestors_and_descendants_by_id(
        concept_id,
        min_separation_ancestor=1, 
        min_separation_descendant=1, 
        max_separation_ancestor=1, 
        max_separation_descendant=1 
    )
    results_refactor = session.execute(query_new).fetchall()
    session.close()
    
    # Convert to comparable format
    results_refactor = [tuple(row) for row in results_refactor]
    
    # Sort both for comparison
    results_original_sorted = sorted(results_original)
    results_refactor_sorted = sorted(results_refactor)

    assert results_original_sorted == results_refactor_sorted


def test_regression_query_related_by_id(db_connection): 
    concept_id = 1125315
    query = f"""
        SELECT
            cr.concept_id_2 AS concept_id,
            cr.concept_id_1,
            cr.relationship_id,
            cr.concept_id_2,
            c.concept_name,
            c.vocabulary_id,
            c.concept_code
        FROM
            {settings.db_schema}.concept_relationship cr
        JOIN
            {settings.db_schema}.concept c ON cr.concept_id_2 = c.concept_id
        WHERE
            cr.concept_id_1 = :concept_id AND
            cr.valid_end_date > NOW()
    """
    
    # Execute original query
    Session = sessionmaker(db_connection)
    session = Session()
    results_original_raw = session.execute(sql.text(query), {"concept_id": concept_id}).fetchall()
    
    # Process original results - remove duplicates and filter out self-references
    seen = set()
    results_original = []
    for row in results_original_raw:
        row_tuple = tuple(row)
        if row_tuple not in seen and row[0] != concept_id:  # concept_id is at index 0
            seen.add(row_tuple)
            results_original.append(row_tuple)
    
    # New query using SQLAlchemy
    query_new = query_related_by_id(concept_id)
    results_refactor = session.execute(query_new).fetchall()
    session.close()

    # Convert to comparable format
    results_refactor = [tuple(row) for row in results_refactor]
    
    # Sort both for comparison
    results_original_sorted = sorted(results_original)
    results_refactor_sorted = sorted(results_refactor)

    assert results_original_sorted == results_refactor_sorted
