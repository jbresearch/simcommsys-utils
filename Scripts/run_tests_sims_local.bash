#!/usr/bin/bash

set -e

SCRIPT_DIR=$(realpath $(dirname ${BASH_SOURCE[0]}))
cd "${SCRIPT_DIR}/.."

echo
echo "TEST CASE 4"
echo "==========="
echo "Testing running jobs in local execution context ..."

poetry run simcommsys-utils run-jobs \
    --config-file TestData/Jobfiles/local.yaml \
    --group group1
poetry run simcommsys-utils run-jobs \
    --config-file TestData/Jobfiles/local.yaml \
    --group group2

echo
echo "TEST CASE 5"
echo "==========="
echo "Testing running jobs in masterslave execution context ..."

poetry run simcommsys-utils run-jobs \
    --config-file TestData/Jobfiles/masterslave.yaml
