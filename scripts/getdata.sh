#!/usr/bin/env bash

# exit on error
set -e

BASE_DIR=$HOME/data
DATA_DIR=$BASE_DIR/twconv/trec
STAGGING_DIR=${DATA_DIR}/datastagging

# download datasets



# tf records builder
python -m twconvrecusers.preprocessing.tfrecords_builder \
   --input_dir=data/convusersec/twconvrsu_csv_v1 \
   --output_dir=data/convusersec/twconvrsu_tf_v1 \
   --max_sentence_len=1400

#python3 -m preprocessing.embeddings_builder \
#    data/convusersec/twconvrsu_tf_v1/vocabulary.txt \
#    embeddings/fasttext/cc.es.300.vec
#
#
#
