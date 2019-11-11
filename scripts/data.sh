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


##################################################################
# TREC dataset

# setup directories
BASE_DIR=$HOME/data
EMBEDDINGS_DIR=${BASE_DIR}/embeddings/fasttext
DATA_DIR=$BASE_DIR/twconv/2011_trec
if [[ -d ${DATA_DIR} ]]; then rm -rf ${DATA_DIR};fi
mkdir -p ${DATA_DIR}
# download file
ZIP_FILE=2011_trec.v6.zip
URL=https://storage.googleapis.com/ml-research-datasets/twconv/${ZIP_FILE}
wget -O ${DATA_DIR}/${ZIP_FILE} $URL
unzip -o ${DATA_DIR}/${ZIP_FILE}  -d ${DATA_DIR}

#DATA_STAGGING=${DATA_DIR}/sampledata
#python -m twconvrecusers.data.tfrecords \
#   --input_dir=${DATA_STAGGING} \
#   --num_distractors=5 \
#   --max_sentence_len=10
#
#DATA_STAGGING=${DATA_DIR}/staggingdata
#mkdir -p ${DATA_STAGGING}
#python -m twconvrecusers.data.tfrecords \
#   --input_dir=${DATA_STAGGING} \
#   --num_distractors=9 \
#   --max_sentence_len=120

#python -m twconvrecusers.data.embeddings \
#    ${DATA_STAGGING}/vocabulary.txt \
#    ${EMBEDDINGS_DIR}/crawl-300d-2M.vec

