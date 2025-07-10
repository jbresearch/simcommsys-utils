#!/usr/bin/bash

set -e

SCRIPT_DIR=`realpath $(dirname ${BASH_SOURCE[0]})`
cd "${SCRIPT_DIR}/.."

echo "TEST CASE 1"
echo "==========="
echo "Testing conversion between parity check matrix formats ..."

poetry run simcommsys-utils convert-pchk \
    --from-format alist --to-format flat \
    --out-delimiter ',' --out-transpose \
    TestData/Codes/H_N_96_M_48.alist TestData/Codes/H_N_96_M_48.csv
echo "Conversion from alist to flat OK."
poetry run simcommsys-utils convert-pchk \
    --from-format flat --to-format simcommsys \
    --in-delimiter ',' --in-transpose --values-method random --random-seed 0 \
    TestData/Codes/H_N_96_M_48.csv TestData/Codes/H_N_96_M_48.txt
echo "Conversion from flat to Simcommsys OK."
poetry run simcommsys-utils check-pchk \
    --format simcommsys \
    TestData/Codes/H_N_96_M_48.txt
echo "Check of Simcommsys format file OK."

echo
echo "TEST CASE 2"
echo "==========="
echo "Building LDPC systems from parity check files and templates ..."
poetry run simcommsys-utils make-ldpc-systems \
    --real-type double --real-type float --gf gf2 --spa-type gdl \
    TestData/Codes/ TestData/Templates/awgn TestData/Systems
echo "Construction of LDPC system files from templates in TestData/Templates/awgn OK."
poetry run simcommsys-utils make-ldpc-systems \
    --real-type float --gf gf2 --gf gf8 --gf gf16 --spa-type gdl_cuda \
    TestData/Codes/ TestData/Templates/qsc TestData/Systems
echo "Construction of LDPC system files from templates in TestData/Templates/qsc OK."

echo
echo "TEST CASE 3"
echo "==========="
echo "Testing generation of simulator and timer files ..."
# NOTE: --simcommsys-tag probably has to be changed on your system.
poetry run simcommsys-utils make-simulators \
    --resultscollector errors_hamming --input-mode random --analyze-decode-iters --simcommsys-tag development-omp-mpi-gmp-cuda89 \
    TestData/Systems TestData/Simulators
echo "Generation of Simcommsys simulator files OK."
poetry run simcommsys-utils make-timers TestData/Systems TestData/Timers
echo "Generation of Simcommsys timer files OK."

echo
echo "TEST CASE 4"
echo "==========="
echo "Testing running jobs in local and master-slave execution contexts ..."

poetry run simcommsys-utils run-jobs \
    --config-file TestData/Jobfiles/jobspec-local-1.yaml \
    --group group1
poetry run simcommsys-utils run-jobs \
    --config-file TestData/Jobfiles/jobspec-local-1.yaml \
    --group group2