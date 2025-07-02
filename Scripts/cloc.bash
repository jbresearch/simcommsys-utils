#!/usr/bin/bash
# Count number of lines in project (just for fun)

set -e

help() {
    echo "Usage: ./cloc.bash [-h]"
    echo "  -h  Print help message and exit"
}

# parse arguments to script
while getopts 'h' opt; do
    case ${opt} in
        h) help
           exit 0 ;;
    esac
done

SCRIPT_DIR=`realpath $(dirname ${BASH_SOURCE[0]})`
BASE_DIR="${SCRIPT_DIR}/.."

cd ${BASE_DIR}
cloc --exclude-dir=.venv,__pycache__,.vscode \
    --exclude-ext=lock \
    --fullpath --not-match-d=TestData --not-match-d=Examples \
    .