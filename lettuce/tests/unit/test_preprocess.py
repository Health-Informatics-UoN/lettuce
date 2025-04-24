import pytest
from omop.preprocess import preprocess_search_term


def test_preprocess_one_word():
    assert preprocess_search_term("paracetamol") == "paracetamol"


def test_preprocess_two_words():
    assert preprocess_search_term("paracetamol caffeine") == "paracetamol | caffeine"


def test_preprocess_with_stopword():
    assert (
        preprocess_search_term("paracetamol and caffeine") == "paracetamol | caffeine"
    )
