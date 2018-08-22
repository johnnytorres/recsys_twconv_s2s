#!/usr/bin/env bash

#profiles

python3 -m preprocessing.profiles_builder clean \
    --input=data/convusersec/timelines_raw.csv

python3 -m preprocessing.profiles_builder build \
    --input=data/convusersec/timelines_raw.csv \
    --output=data/convusersec/timelines.csv \
    --n=10


#conversations

# trainset
python3 -m preprocessing.csv_builder \
    --data-root=data/convusersec/dialogs \
    --profiles-path=data/convusersec/timelines.csv \
    --output=data/convusersec/twconvrsu_csv_v1/train.csv \
    --text-field=2 \
    --min-context-length=2 \
    --tokenize-punk \
    -e=50000 \
    train

# valid set
python3 -m preprocessing.csv_builder \
    --data-root=data/convusersec/dialogs \
    --profiles-path=data/convusersec/timelines.csv \
    --output=data/convusersec/twconvrsu_csv_v1/valid.csv \
    --text-field=2 \
    --min-context-length=2 \
    --tokenize-punk \
    -e=5000 \
    valid

#test
python3 -m preprocessing.csv_builder \
    --data-root=data/convusersec/dialogs \
    --profiles-path=data/convusersec/timelines.csv \
    --output=data/convusersec/twconvrsu_csv_v1/test.csv \
    --text-field=2 \
    --min-context-length=2 \
    --tokenize-punk \
    -e=5000 \
    test

# tf records builder
python3 -m preprocessing.tfrecords_builder \
    --input_dir=data/convusersec/twconvrsu_csv_v1 \
    --output_dir=data/convusersec/twconvrsu_tf_v1 \
    --max_sentence_len=1400

python3 -m preprocessing.embeddings_builder \
    data/convusersec/twconvrsu_tf_v1/vocabulary.txt \
    embeddings/fasttext/cc.es.300.vec



