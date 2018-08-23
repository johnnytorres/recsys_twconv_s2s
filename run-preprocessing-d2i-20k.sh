#!/usr/bin/env bash

# profiles

python3 -m preprocessing.profiles_builder \
    --input=data/convusersec/timelines_raw.csv \
    --output=data/convusersec/timelines.csv \
    --tokenize-tweets \
    --n=10

#conversations
# "dialogs" dataset is provided by crawler

#train
python3 -m preprocessing.csv_builder \
    --data-root=data/convusersec/dialogs \
    --profiles-path=data/convusersec/timelines.csv \
    --output=data/convusersec/twconvrsu_csv_v2i_20k/train.csv \
    --text-field=2 \
    --min-context-length=2 \
    --tokenize-tweets \
    -e=20000 \
    train

#valid
python3 -m preprocessing.csv_builder \
    --data-root=data/convusersec/dialogs \
    --profiles-path=data/convusersec/timelines.csv \
    --output=data/convusersec/twconvrsu_csv_v2i_20k/valid.csv \
    --text-field=2 \
    --min-context-length=2 \
    --tokenize-tweets \
    -e=5000 \
    valid

#test
python3 -m preprocessing.csv_builder \
    --data-root=data/convusersec/dialogs \
    --profiles-path=data/convusersec/timelines.csv \
    --output=data/convusersec/twconvrsu_csv_v2i_20k/test.csv \
    --text-field=2 \
    --min-context-length=2 \
    --tokenize-tweets \
    -e=5000 \
    test

# tf records builder
python3 -m preprocessing.tfrecords_builder \
    --input_dir=data/convusersec/twconvrsu_csv_v2i_20k \
    --output_dir=data/convusersec/twconvrsu_tf_v2i_20k \
    --max_sentence_len=1400

python3 -m preprocessing.embeddings_builder \
    data/convusersec/twconvrsu_tf_v2i_20k/vocabulary.txt \
    embeddings/fasttext/cc.es.300.vec



