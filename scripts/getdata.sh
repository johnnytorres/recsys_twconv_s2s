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

# PREPARE DATASETS
# TREC dataset
DATA_DIR=$BASE_DIR/twconv/trec
mkdir -p ${DATA_DIR}
URL=https://storage.googleapis.com/ml-research-datasets/convai/trec.zip
wget -O ${DATA_DIR}/trec.zip $URL
# TODO: download other datasets

# tf records builder

DATA_STAGGING=${DATA_DIR}/sampledata
python -m twconvrecusers.data.tfrecords_builder \
   --input_dir=${DATA_STAGGING} \
   --num_distractors=5 \
   --max_sentence_len=10


DATA_STAGGING=${DATA_DIR}/staggingdata
python -m twconvrecusers.data.tfrecords_builder \
   --input_dir=${DATA_STAGGING} \
   --num_distractors=9 \
   --max_sentence_len=160


