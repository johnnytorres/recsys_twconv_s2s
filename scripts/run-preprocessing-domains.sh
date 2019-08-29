#!/usr/bin/env bash


#conversations
# "dialogs" datasets is provided by crawler

#train
PROFILES_PATH=datasets/convusersec/timelines.csv
DOMAIN=politics # change to politics, sports, activism
DATA_DIR=datasets/convusersec/dialogs_$DOMAIN
CSV_DIR=datasets/convusersec/twconvrsu_csv_$DOMAIN
TF_DIR=datasets/convusersec/twconvrsu_tf_$DOMAIN

mkdir $CSV_DIR
mkdir $TF_DIR

python3 -m datasets.csv_builder \
    --datasets-root=$DATA_DIR \
    --profiles-path=$PROFILES_PATH \
    --output=$CSV_DIR/train.csv \
    --text-field=2 \
    --min-context-length=2 \
    --tokenize-tweets \
    -e=1000 \
    train

#valid
python3 -m datasets.csv_builder \
    --datasets-root=$DATA_DIR \
    --profiles-path=$PROFILES_PATH \
    --output=$CSV_DIR/valid.csv \
    --text-field=2 \
    --min-context-length=2 \
    --tokenize-tweets \
    -e=5000 \
    valid

#tests
python3 -m datasets.csv_builder \
    --datasets-root=$DATA_DIR \
    --profiles-path=$PROFILES_PATH \
    --output=$CSV_DIR/tests.csv \
    --text-field=2 \
    --min-context-length=2 \
    --tokenize-tweets \
    -e=5000 \
    tests

# tf records builder
python3 -m datasets.tfrecords_builder \
    --input_dir=$CSV_DIR \
    --output_dir=$TF_DIR \
    --max_sentence_len=1400

#python3 -m datasets.embeddings_builder \
#    $TF_DIR/vocabulary.txt \
#    embeddings/fasttext/cc.es.300.vec



