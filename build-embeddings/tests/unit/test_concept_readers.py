import pytest

from embedding_utils.concept_readers import CsvConceptExtractor, parse_rows
from embedding_utils.string_building import Concept

@pytest.fixture
def dummy_rows() -> list[tuple[int, str, str, str, str]]:
    return [
            (12345, "This isn't real", "Made up", "gobbledegook", "totally fictional"),
            (56789, "This one too", "ersatz", "mumbo jumbo", "unreal")
            ]

def test_parse_rows(dummy_rows):
    concepts = parse_rows(dummy_rows)

    assert isinstance(concepts, list)
    assert len(concepts) == 2
    assert isinstance(concepts[0], Concept)
    assert concepts[0] == Concept(
            concept_id=12345,
            concept_name="This isn't real",
            domain="Made up",
            vocabulary="gobbledegook",
            concept_class="totally fictional"
            )

CSV_CONTENTS = """concept_id\tconcept_name\tdomain_id\tvocabulary_id\tconcept_class_id\tstandard_concept\tconcept_code\tvalid_start_date\tvalid_end_date\tinvalid_reason
12345\tThis isn't real\tMade up\tgobbledegook\ttotally fictional\tS\tquijibo\tnow\tforever\tnone
56789\tThis one too\tersatz\tmumbo jumbo\tunreal\tS\tquijibo\tnow\tforever\tnone
19238579\tAnother one\tersatz\tmumbo jumbo\tunreal\tS\tquijibo\tnow\tforever\tnone"""


@pytest.fixture
def mock_csv_extractor(tmp_path) -> CsvConceptExtractor:
    return CsvConceptExtractor(
            path=tmp_path / "file.csv",
            batch_size=2
            )

def test_csv_extractor(tmp_path, mock_csv_extractor):
    temp_csv = tmp_path / "file.csv"

    with open(temp_csv, "w") as f:
        f.write(CSV_CONTENTS)

    concepts_batches = list(mock_csv_extractor.load_concept_batch())
    
    assert isinstance(concepts_batches, list)
    assert len(concepts_batches) ==  2
    first_batch = concepts_batches[0]
    assert isinstance(first_batch[0], Concept)
    assert first_batch[0] == Concept(
            concept_id=12345,
            concept_name="This isn't real",
            domain="Made up",
            vocabulary="gobbledegook",
            concept_class="totally fictional"
            )

    concepts_all = list(mock_csv_extractor.load_concepts())

    assert isinstance(concepts_all, list)
    assert len(concepts_all) ==  3
    assert isinstance(concepts_all[0], Concept)
    assert concepts_all[0] == Concept(
            concept_id=12345,
            concept_name="This isn't real",
            domain="Made up",
            vocabulary="gobbledegook",
            concept_class="totally fictional"
            )
