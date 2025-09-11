import os
from options.base_options import BaseOptions
import pytest
import pandas as pd 
from sqlalchemy import create_engine
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
    
    columns = [
        'relationship_type', 'concept_id', 'ancestor_concept_id',
        'descendant_concept_id', 'concept_name', 'vocabulary_id',
        'concept_code', 'min_levels_of_separation', 'max_levels_of_separation'
    ]
    results = pd.DataFrame(results, columns=columns)
    names = results["concept_name"].to_list()

    assert results["relationship_type"].unique().tolist() == ["Ancestor", "Descendant"]
    assert "Painaid BRF Oral Product" in names
    assert "homatropine methylbromide; systemic" in names 


def test_query_related_by_id(db_connection):
    concept_id = 1125315  # Acetaminophen

    Session = sessionmaker(db_connection)
    session = Session()

    query = query_related_by_id(concept_id)
    results_refactor = session.execute(query).fetchall()
    columns = [
        'concept_id', 'concept_id_1', 'relationship_id', 'concept_id_2',
        'concept_name', 'vocabulary_id', 'concept_code'
    ]
    results = pd.DataFrame(results_refactor, columns=columns)

    assert len(results) >= 1

    names = results["concept_name"].to_list()
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
                    ca.descendant_concept_id = %s AND
                    ca.min_levels_of_separation >= %s AND
                    ca.max_levels_of_separation <= %s
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
                    ca.ancestor_concept_id = %s AND
                    ca.min_levels_of_separation >= %s AND
                    ca.max_levels_of_separation <= %s
            )
        """
    concept_id = 1125315
    min_separation_ancestor = 1
    min_separation_descendant = 1
    max_separation_ancestor = 1
    max_separation_descendant = 1


    params = (
        concept_id,
        min_separation_ancestor,
        max_separation_ancestor,
        concept_id,
        min_separation_descendant,
        max_separation_descendant,
    )
    results_original = (
        pd.read_sql(query, con=db_connection, params=params)
        .drop_duplicates()
        .query("concept_id != @concept_id")
    )

    # New query using SQLAlchemy
    Session = sessionmaker(db_connection)
    session = Session()

    query = query_ancestors_and_descendants_by_id(
        concept_id,
        min_separation_ancestor = 1, 
        min_separation_descendant = 1, 
        max_separation_ancestor = 1, 
        max_separation_descendant = 1 
    )
    results = session.execute(query).fetchall()
    session.close()

    columns = [
        'relationship_type', 'concept_id', 'ancestor_concept_id',
        'descendant_concept_id', 'concept_name', 'vocabulary_id',
        'concept_code', 'min_levels_of_separation', 'max_levels_of_separation'
    ]
    results_refactor = pd.DataFrame(results, columns=columns)
    
    results_original_sorted = results_original.sort_values(by=columns).reset_index(drop=True)
    results_refactor_sorted = results_refactor.sort_values(by=columns).reset_index(drop=True)

    assert results_original_sorted.equals(results_refactor_sorted)


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
            cr.concept_id_1 = %s AND
            cr.valid_end_date > NOW()
    """
    # Old query 
    results_original = (
        pd.read_sql(query, con=db_connection, params=(concept_id,))
        .drop_duplicates()
        .query("concept_id != @concept_id")
    )

    # New query using SQLAlchemy
    Session = sessionmaker(db_connection)
    session = Session()

    query = query_related_by_id(concept_id)
    results_refactor = session.execute(query).fetchall()
    session.close()

    columns = [
        'concept_id', 'concept_id_1', 'relationship_id', 'concept_id_2',
        'concept_name', 'vocabulary_id', 'concept_code'
    ]
    results_refactor = pd.DataFrame(results_refactor, columns=columns)

    results_original_sorted = results_original.sort_values(by=columns).reset_index(drop=True)
    results_refactor_sorted = results_refactor.sort_values(by=columns).reset_index(drop=True)

    assert results_original_sorted.equals(results_refactor_sorted)
