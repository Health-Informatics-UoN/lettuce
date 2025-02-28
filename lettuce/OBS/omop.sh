# Script to set up OMOP database on local machine - Test Purpose

#!/bin/bash
MYSQL_ROOT_PASSWORD=051000
BASIC_AUTH_USERNAME=APIuser
BASIC_AUTH_PASSWORD=psw4API
OMOP_DATA_FOLDER=data

echo "Using local database"
# Linux
# apt-get update && apt-get install -y mariadb-server dialog

# MacOS
brew update
brew install mariadb

# Linux
# Start MySQL server manually
# mysqld_safe & 
# sleep 10

# MacOS
brew services start mariadb
sleep 10

# Set up MySQL
mysql -u root -e "GRANT ALL ON *.* TO 'root'@'localhost' IDENTIFIED BY '${MYSQL_ROOT_PASSWORD}'; FLUSH PRIVILEGES;"

# Increase MySQL timeouts
mysql -u root -p${MYSQL_ROOT_PASSWORD} -e "SET GLOBAL wait_timeout=28800; SET GLOBAL interactive_timeout=28800;"

# Run initial script
echo "Running init_db.sql"
mysql -u root -p${MYSQL_ROOT_PASSWORD} < ./init_db.sql

# Function to process TSV files
process_tsv() {
    table=$1
    file=$2

    full_path=$(realpath $file)

    echo "Importing $file into $table"
    mysql -u root -p${MYSQL_ROOT_PASSWORD} mydb -e "
    LOAD DATA LOCAL INFILE '$full_path'
    INTO TABLE $table
    FIELDS TERMINATED BY '\t'
    LINES TERMINATED BY '\n'
    IGNORE 1 ROWS;
    "
}
export -f process_tsv

# Process file import
for table in CONCEPT CONCEPT_SYNONYM CONCEPT_ANCESTOR CONCEPT_RELATIONSHIP; 
    do
    echo "Processing $table files"
    for file in ./${OMOP_DATA_FOLDER}/$table/*.tsv; 
        do
        process_tsv $table $file
        done
    done

# Create indexes for CONCEPT_RELATIONSHIP
echo "Indexing CONCEPT_RELATIONSHIP in Database"
mysql -u root -p${MYSQL_ROOT_PASSWORD} mydb -e "
CREATE INDEX idx_concept_relationship_on_id1_enddate ON CONCEPT_RELATIONSHIP(concept_id_1, valid_end_date);
CREATE INDEX idx_concept_relationship_on_id2 ON CONCEPT_RELATIONSHIP(concept_id_2);
    SELECT COUNT(*) AS 'Number of rows in CONCEPT_RELATIONSHIP table' FROM CONCEPT_RELATIONSHIP;
"

# Create indexes for CONCEPT_ANCESTOR
echo "Indexing CONCEPT_ANCESTOR in Database"
mysql -u root -p${MYSQL_ROOT_PASSWORD} mydb -e "
CREATE INDEX idx_concept_ancestor_on_descendant_id ON CONCEPT_ANCESTOR(descendant_concept_id);
CREATE INDEX idx_concept_ancestor_on_ancestor_id ON CONCEPT_ANCESTOR(ancestor_concept_id);
SELECT COUNT(*) AS 'Number of rows in CONCEPT_ANCESTOR table' FROM CONCEPT_ANCESTOR;
"

# Create Fulltext index and count rows for CONCEPT_SYNONYM
echo "Indexing CONCEPT_SYNONYM in Database" 
mysql -u root -p${MYSQL_ROOT_PASSWORD} mydb -e "
CREATE FULLTEXT INDEX idx_concept_synonym_name ON CONCEPT_SYNONYM(concept_synonym_name);
CREATE INDEX idx_concept_synonym_id ON CONCEPT_SYNONYM (concept_id);
SELECT COUNT(*) AS 'Number of rows in CONCEPT_SYNONYM table' FROM CONCEPT_SYNONYM;
"

# Create Fulltext index and count rows for CONCEPT
echo "Indexing CONCEPT in Database" 
mysql -u root -p${MYSQL_ROOT_PASSWORD} mydb -e "
CREATE FULLTEXT INDEX idx_concept_name ON CONCEPT(concept_name);
CREATE INDEX idx_concept_id ON CONCEPT (concept_id);
CREATE INDEX idx_vocabulary_id ON CONCEPT (vocabulary_id);
CREATE INDEX idx_standard_concept ON CONCEPT (standard_concept);
SELECT COUNT(*) AS 'Number of rows in CONCEPT table' FROM CONCEPT;
"