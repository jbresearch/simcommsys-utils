#!/usr/bin/bash

set -e

SCRIPT_DIR=$(realpath $(dirname ${BASH_SOURCE[0]}))
cd "${SCRIPT_DIR}/.."

echo
echo "TEST CASE 6"
echo "==========="
echo "Testing running jobs in SLURM execution context ..."

poetry run simcommsys-utils run-jobs \
    --config-file TestData/Jobfiles/slurm.yaml \
    --group group1
poetry run simcommsys-utils run-jobs \
    --config-file TestData/Jobfiles/slurm.yaml \
    --group group2

echo
echo "TEST CASE 7"
echo "==========="
echo "Testing running jobs in SLURM masterslave execution context ..."

poetry run simcommsys-utils run-jobs \
    --config-file TestData/Jobfiles/slurm_masterslave.yaml
