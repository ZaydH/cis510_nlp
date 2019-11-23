#!/usr/bin/env bash

DEFAULT_PARTITION=gpu
TEST_PARTITION=test
LONG_PARTITION=long

if [ $# -ge 2 ]; then
    PROG_NAME=$( basename $0 )
    printf "Expected Call: ${PROG_NAME} [${LONG_PARTITION}|${TEST_PARTITION}]\n\n"
    printf "Requests a GPU bash terminal for debug\n"
    printf "Select the GPU partition. Default is \"${DEFAULT_PARTITION}\".  Otherwise select \"${LONG_PARTITION}${DEFAULT_PARTITION}\" or \"${TEST_PARTITION}${DEFAULT_PARTITION}\"\n"
    exit 1
fi

PARTITION=${DEFAULT_PARTITION}  # Default if nothing specified
TIME_LIMIT="1-00:00:00"  # 1 day
if [ $# -eq 1 ]; then
    PART_PREFIX=$1
    if [ ${PART_PREFIX} == ${TEST_PARTITION} ]; then
        TIME_LIMIT="4:00:00"
    elif [ ${PART_PREFIX} == ${LONG_PARTITION} ]; then
        TIME_LIMIT="14-00:00:00"  # 14 days
    else
        printf "Unsupported GPU partition prefix: \"${PART_PREFIX}\"\n"
        printf "Valid options are \"${LONG_PARTITION}\" and \"${TEST_PARTITION}\"\n"
        exit 1
    fi
    PARTITION=${PART_PREFIX}${DEFAULT_PARTITION}
fi

printf "Partition Requested: ${PARTITION}\n"
printf "Time Limit Requested: ${TIME_LIMIT}\n"
# Get a bash console with one GPU
srun -A uoml --pty --gres=gpu:1 --cpus-per-gpu=2 --mem=128G --time=${TIME_LIMIT} --partition=${PARTITION} bash
