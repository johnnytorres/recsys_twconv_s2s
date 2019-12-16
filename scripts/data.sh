#!/usr/bin/env bash

# exit on error
set -e

BASE_DIR=$HOME/data

# PREPARE EMBEDDINGS
EMBEDDINGS_DIR=${BASE_DIR}/embeddings/fasttext
mkdir -p ${EMBEDDINGS_DIR}
# fasttext english download
EMBEDDINGS_FILE=crawl-300d-2M.vec
if [[ ! -f ${EMBEDDINGS_FILE} ]]
then
    echo "downloading fasttext embeddings..."
    URL=https://dl.fbaipublicfiles.com/fasttext/vectors-english/${EMBEDDINGS_FILE}.zip
    wget -O ${EMBEDDINGS_DIR}/${EMBEDDINGS_FILE}.zip $URL
    echo "decompressing fasttext embeddings..."
    unzip -d ${EMBEDDINGS_DIR} ${EMBEDDINGS_FILE}.zip
fi

