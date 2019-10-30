#!/usr/bin/env zsh

if [ $# -ne 3 ]; then
    printf "Invalid input arguments.\n"
    FILE_NAME=$(basename $0)
    printf "Expected call: ${FILE_NAME} <train_file> <test_file> <key_file>\n"
    exit 1
fi

TRAIN_FILE=$1
TEST_FILE=$2
OUT_FILE=out_test.txt

rm -rf ${OUT_FILE} > /dev/null
python viterbi.py ${TRAIN_FILE} ${TEST_FILE} ${OUT_FILE} --smooth

KEY_FILE=$3
python scorer.py ${KEY_FILE} ${OUT_FILE}
