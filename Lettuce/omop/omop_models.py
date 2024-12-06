from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String


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
