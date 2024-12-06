from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Date, Integer, String


from dotenv import load_dotenv
from os import environ


load_dotenv()
DB_SCHEMA = environ["DB_SCHEMA"]

Base = declarative_base()


class Concept(Base):
    """
    This class represents an ORM mapping to the OMOP concept table
    """

    __tablename__ = "concept"
    __table_args__ = {"schema": DB_SCHEMA}

    concept_id = Column(Integer, primary_key=True)
    concept_name = Column(String)
    vocabulary_id = Column(String)
    concept_code = Column(String)
    standard_concept = Column(String)

    def __repr__(self) -> str:
        return super().__repr__()


class ConceptSynonym(Base):
    """
    This class represents an ORM mapping to the OMOP concept_synonym table
    """

    __tablename__ = "concept_synonym"
    __table_args__ = {"schema": DB_SCHEMA}

    concept_id = Column(Integer, primary_key=True)
    concept_synonym_name = Column(String)
    language_concept_id = Column(Integer)

    def __repr__(self) -> str:
        return super().__repr__()


class ConceptRelationship(Base):
    """
    This class represents an ORM mapping to the OMOP concept_relationship table
    """

    __tablename__ = "concept_relationship"
    __table_args__ = {"schema": DB_SCHEMA}

    concept_id_1 = Column(Integer)
    concept_id_2 = Column(Integer)
    relationship_id = Column(String)
    valid_start_date = Column(Date)
    valid_end_date = Column(Date)
    invalid_reason = Column(String)
    # Inserting dummy primary key in ORM layer
    dummy_primary = Column(Integer, primary_key=True)

    def __repr__(self) -> str:
        return super().__repr__()
