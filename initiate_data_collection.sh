#!/bin/bash

mkdir -p data

curl -L -o data/chembl_36_sqlite.tar.gz "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_36_sqlite.tar.gz"

tar -xzvf data/chembl_36_sqlite.tar.gz -C data

rm data/chembl_36_sqlite.tar.gz