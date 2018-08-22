#!/usr/bin/env bash


DATA_DIR=~/data/ubuntu/

python3 -m csv_builder \
    --data-root=${DATA_DIR}/dialogs \
    --output=${DATA_DIR}/train.csv \
    --tokenize-punk \
    -e=1000000 \
    train

python3 -m csv_builder \
    --data-root=${DATA_DIR}/dialogs \
    --output=${DATA_DIR}/valid.csv \
    --tokenize-punk \
    valid

python3 -m csv_builder \
    --data-root=${DATA_DIR}/dialogs \
    --output=${DATA_DIR}/test.csv \
    --tokenize-punk \
    test

python3 -m tfrecords_builder \
    --input_dir=${DATA_DIR} \
    --output_dir=${DATA_DIR}/big_tftp

#python3 -m preprocessing.tfrecords_builder \
#    --input_dir=${DATA_DIR} \
#    --output_dir=${DATA_DIR}/big_tftp \
#    --example