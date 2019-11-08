#!/usr/bin/env zsh

HW=hw03
SUBMISSION_DIR=$(pwd)/${HW}

TEX_DIR=$(pwd)/tex
TEX_BASE=${HW}

SRC_DIR=$(pwd)/src

function build_submission_dir() {
    printf "Deleting and recreating submission directory..."
    rm -rf ${SUBMISSION_DIR}
    mkdir -p ${SUBMISSION_DIR}
    printf "COMPLETED\n"
}


function build_src() {
    DATASET_DIR=datasets

    DRIVER_FILE=max_ent_trainer.py
    TRAIN_FILE=${DATASET_DIR}/train.pos-chunk-name
    TEST_FILE=${DATASET_DIR}/test.pos-chunk
    OUT_FILE=${DATASET_DIR}/test.name

    DATASETS_DIR=datasets
    FEATURE_INFO_DIR=feature_info
    JAVA_DIR=maxent

    cd ${SRC_DIR} > /dev/null

    printf "Running Maximum Entropy algorithm..."
    rm -rf ${OUT_FILE} > /dev/null
    python ${DRIVER_FILE} ${TRAIN_FILE} ${TEST_FILE} ${OUT_FILE} > /dev/null
    printf "COMPLETED\n"

    printf "Copying source files..."
    OUT_SRC_DIR=${SUBMISSION_DIR}/src
    rm -rf ${OUT_SRC_DIR} > /dev/null
    mkdir -p ${OUT_SRC_DIR} > /dev/null

    # Copy Python source code
    cp *.py ${OUT_SRC_DIR} > /dev/null
    # Copy label file
    cp ${OUT_FILE} ${SUBMISSION_DIR} > /dev/null

    # Copy the Java files
    OUT_JAVA_DIR=${OUT_SRC_DIR}/${JAVA_DIR}
    mkdir -p ${OUT_JAVA_DIR} > /dev/null
    cp ${JAVA_DIR}/*.java ${OUT_JAVA_DIR} > /dev/null
    cp ${JAVA_DIR}/*.jar ${OUT_JAVA_DIR} > /dev/null

    # Copy the dataset files
    cp -r ${DATASETS_DIR} ${OUT_SRC_DIR} > /dev/null

    # Copy the feature information files
    cp -r ${FEATURE_INFO_DIR} ${OUT_SRC_DIR} > /dev/null
    printf "COMPLETED\n"
    cd - > /dev/null
}


function build_tex() {
    cd ${TEX_DIR} > /dev/null
    printf "Clearing existing TeX files..."
    # Clear LaTeX temp files
    latexmk -C &> /dev/null
    # Remove the PDF for good measure
    rm -rf ${TEX_BASE}.pdf > /dev/null
    printf "COMPLETED\n"
    printf "Compiling the LaTeX submission..."
    pdflatex ${TEX_BASE}.tex > /dev/null
    # May take multiple compilations to ensure that all references properly built
    pdflatex ${TEX_BASE}.tex > /dev/null
    pdflatex ${TEX_BASE}.tex > /dev/null
    cp ${TEX_BASE}.pdf ${SUBMISSION_DIR} > /dev/null
    printf "COMPLETED\n"
    cd -
}

function zip_submission() {
    printf "Zipping submission..."
    ZIP_FILE=zayd_hammoudeh_${HW}.zip
    rm -rf ${ZIP_FILE} > /dev/null

    zip -r ${ZIP_FILE} $(realpath --relative-to=. ${SUBMISSION_DIR}) > /dev/null
    rm -rf ${SUBMISSION_DIR} > /dev/null
    unzip ${ZIP_FILE} > /dev/null
    printf "COMPLETED\n"
}

build_submission_dir
build_src
build_tex
zip_submission
