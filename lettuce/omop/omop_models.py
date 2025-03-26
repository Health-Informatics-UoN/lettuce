from sqlalchemy.orm import declarative_base, mapped_column
from sqlalchemy import Column, Date, Integer, String
from pgvector.sqlalchemy import Vector

from os import environ

DB_SCHEMA = environ["DB_SCHEMA"]
DB_VECTABLE = environ["DB_VECTABLE"]
DB_VECSIZE = int(environ["DB_VECSIZE"])

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


class ConceptAncestor(Base):
    """
    This class represents an ORM mapping to the OMOP concept_ancestor table
    """

    __tablename__ = "concept_ancestor"
    __table_args__ = {"schema": DB_SCHEMA}

    ancestor_concept_id = Column(Integer)
    descendant_concept_id = Column(Integer)
    min_levels_of_separation = Column(Integer)
    max_levels_of_separation = Column(Integer)

    dummy_primary = Column(Integer, primary_key=True)

class Embedding(Base):
    """
    This class represents an ORM mapping to an embeddings table
    """

    __tablename__ = DB_VECTABLE
    __table_args__ = {"schema": DB_SCHEMA}

    concept_id = Column(Integer)
    embedding = mapped_column(Vector(DB_VECSIZE))
    dummy_primary = Column(Integer, primary_key=True)
