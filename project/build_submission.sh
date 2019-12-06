#!/usr/bin/env zsh

SUBMISSION_NAME=project
SUBMISSION_DIR=$(pwd)/zayd_hammoudeh_${SUBMISSION_NAME}

TEX_DIR=$(pwd)/tex
TEX_BASE=${SUBMISSION_NAME}
TEX_FILE=${SUBMISSION_NAME}.tex

SRC_DIR=$(pwd)/src

function build_submission_dir() {
    printf "Deleting and recreating submission directory..."
    rm -rf ${SUBMISSION_DIR} > /dev/null
    mkdir -p ${SUBMISSION_DIR} > /dev/null
    printf "COMPLETED\n"
}


function build_src() {
    printf "Copying the source directory..."
    cp -r ${SRC_DIR} ${SUBMISSION_DIR} > /dev/null
    printf "COMPLETED\n"

    printf "Deleting files that should not be in submission source code..."
    cd ${SUBMISSION_DIR}/$(basename ${SRC_DIR}) > /dev/null
    ITEMS_TO_DELETE=(".idea" ".data" "tensors" "tb" "__pycache__" "logs" "results" "models")
    for DEL_OBJ in "${ITEMS_TO_DELETE[@]}"; do
        rm -rf ${DEL_OBJ} > /dev/null
    done
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
    latexmk ${TEX_FILE} &> /dev/null

    if [ $? -ne 0 ]; then
        printf "\n"
        msg="ERROR: Building ${TEX_FILE}"
        printf "\e[01;31m${msg}\n\e[0m" >&2
    else
        cp ${TEX_BASE}.pdf ${SUBMISSION_DIR}/zayd_hammoudeh_${SUBMISSION_NAME}.pdf > /dev/null
        printf "COMPLETED\n"
    fi
    cd - > /dev/null
}

function zip_submission() {
    printf "Zipping submission..."
    ZIP_FILE=zayd_hammoudeh_${SUBMISSION_NAME}.zip
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
