#!/usr/bin/env bash

# exit on error
set -e

BASE_DIR=$HOME/data

DATASET=trec
DATA_DIR=$BASE_DIR/twconv/$DATASET
DIALOGS_DIR=$DATA_DIR/dialogs
STAGGING_DIR=${DATA_DIR}/stagging

# download datasets


# it's not necessary as scripts should be run from root project dir

python -m twconvrecusers.preprocessing.tweet_tokenizer \
    --input-file=${DIALOGS_DIR}/dialogs.csv \
    --output-file=${STAGGING_DIR}/dialogs.csv \
    --sep=, \
    --text-field=text \
    --language=english \
    --tokenizer=5 \
    --use-lowercase


# TODO: in case we crawl users' timelines we can build profiles
# python -m twconvrecusers.preprocessing.profiles_builder \
#     --input=$DIALOGS_DIR \
#     --tokenize-tweets \
#     --lowercase \
#     buildfromconv

python -m twconvrecusers.preprocessing.dialogs_builder \
    --data-dir=${DIALOGS_DIR}
    #TODO: --seed for reproducibility

#conversations

## trainset
#python3 -m preprocessing.csv_builder \
#    --data-root=data/convusersec/dialogs \
#    --profiles-path=data/convusersec/timelines.csv \
#    --output=data/convusersec/twconvrsu_csv_v1/train.csv \
#    --text-field=2 \
#    --min-context-length=2 \
#    --tokenize-punk \
#    -e=50000 \
#    train
#
## valid set
#python3 -m preprocessing.csv_builder \
#    --data-root=data/convusersec/dialogs \
#    --profiles-path=data/convusersec/timelines.csv \
#    --output=data/convusersec/twconvrsu_csv_v1/valid.csv \
#    --text-field=2 \
#    --min-context-length=2 \
#    --tokenize-punk \
#    -e=5000 \
#    valid
#
##test
#python3 -m preprocessing.csv_builder \
#    --data-root=data/convusersec/dialogs \
#    --profiles-path=data/convusersec/timelines.csv \
#    --output=data/convusersec/twconvrsu_csv_v1/test.csv \
#    --text-field=2 \
#    --min-context-length=2 \
#    --tokenize-punk \
#    -e=5000 \
#    test
#
## tf records builder
#python3 -m preprocessing.tfrecords_builder \
#    --input_dir=data/convusersec/twconvrsu_csv_v1 \
#    --output_dir=data/convusersec/twconvrsu_tf_v1 \
#    --max_sentence_len=1400
#
#python3 -m preprocessing.embeddings_builder \
#    data/convusersec/twconvrsu_tf_v1/vocabulary.txt \
#    embeddings/fasttext/cc.es.300.vec
#
#
#
