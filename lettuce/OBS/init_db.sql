-- Create DATABASE
CREATE DATABASE IF NOT EXISTS mydb;
USE mydb;

-- Create CONCEPT TABLE
CREATE TABLE IF NOT EXISTS CONCEPT (
    concept_id INT PRIMARY KEY,
    concept_name VARCHAR(500),
    domain_id VARCHAR(20),
    vocabulary_id VARCHAR(20),
    concept_class_id VARCHAR(20),
    standard_concept VARCHAR(1),
    concept_code VARCHAR(50),
    valid_start_date DATE,
    valid_end_date DATE,
    invalid_reason VARCHAR(1)
);

CREATE TABLE IF NOT EXISTS CONCEPT_SYNONYM (
    concept_id INT,
    concept_synonym_name VARCHAR(1000) COLLATE utf8_bin,
    language_concept_id INT,
    PRIMARY KEY (concept_id, concept_synonym_name, language_concept_id),
    FOREIGN KEY (concept_id) REFERENCES CONCEPT(concept_id),
    FOREIGN KEY (language_concept_id) REFERENCES CONCEPT(concept_id)
);

SELECT 'Table CONCEPT_SYNONYM created' AS message;

SELECT 'Table CONCEPT created' AS message;

CREATE TABLE IF NOT EXISTS CONCEPT_RELATIONSHIP (
    concept_id_1 INT,
    concept_id_2 INT,
    relationship_id VARCHAR(20),
    valid_start_date DATE,
    valid_end_date DATE,
    invalid_reason VARCHAR(1),
    PRIMARY KEY (concept_id_1, concept_id_2, relationship_id),
    FOREIGN KEY (concept_id_1) REFERENCES CONCEPT(concept_id),
    FOREIGN KEY (concept_id_2) REFERENCES CONCEPT(concept_id)
);

SELECT 'Table CONCEPT_RELATIONSHIP created' AS message;

CREATE TABLE IF NOT EXISTS CONCEPT_ANCESTOR (
    ancestor_concept_id INT,
    descendant_concept_id INT,
    min_levels_of_separation INT,
    max_levels_of_separation INT,
    PRIMARY KEY (ancestor_concept_id, descendant_concept_id),
    FOREIGN KEY (ancestor_concept_id) REFERENCES CONCEPT(concept_id),
    FOREIGN KEY (descendant_concept_id) REFERENCES CONCEPT(concept_id)
);

SELECT 'Table CONCEPT_ANCESTOR created' AS message;