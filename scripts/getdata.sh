#!/usr/bin/env bash

# exit on error
set -e

BASE_DIR=$HOME/data
<<<<<<< HEAD
DATA_DIR=$BASE_DIR/twconv/trec
=======

DATASET=trec
DATA_DIR=$BASE_DIR/twconv/$DATASET
DIALOGS_DIR=$DATA_DIR/dialogs
>>>>>>> a3c530d9bc372fd6db73a8da3c165b5132deae23
STAGGING_DIR=${DATA_DIR}/datastagging

# download datasets

<<<<<<< HEAD
=======

# it's not necessary as scripts should be run from root project dir

>>>>>>> a3c530d9bc372fd6db73a8da3c165b5132deae23
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
