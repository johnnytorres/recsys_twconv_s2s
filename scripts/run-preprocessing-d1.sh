#!/usr/bin/env bash

#profiles

python3 -m datasets.profiles_builder \
    --input=datasets/convusersec/timelines_raw.csv \
    --output=datasets/convusersec/timelines.csv \
    --n=10


#conversations

# trainset
python3 -m datasets.csv_builder \
    --datasets-root=datasets/convusersec/dialogs \
    --profiles-path=datasets/convusersec/timelines.csv \
    --output=datasets/convusersec/twconvrsu_csv_v1/train.csv \
    --text-field=2 \
    --min-context-length=2 \
    --tokenize-punk \
    -e=50000 \
    train

# valid set
python3 -m datasets.csv_builder \
    --datasets-root=datasets/convusersec/dialogs \
    --profiles-path=datasets/convusersec/timelines.csv \
    --output=datasets/convusersec/twconvrsu_csv_v1/valid.csv \
    --text-field=2 \
    --min-context-length=2 \
    --tokenize-punk \
    -e=5000 \
    valid

#tests
python3 -m datasets.csv_builder \
    --datasets-root=datasets/convusersec/dialogs \
    --profiles-path=datasets/convusersec/timelines.csv \
    --output=datasets/convusersec/twconvrsu_csv_v1/tests.csv \
    --text-field=2 \
    --min-context-length=2 \
    --tokenize-punk \
    -e=5000 \
    tests

# tf records builder
python3 -m datasets.tfrecords_builder \
    --input_dir=datasets/convusersec/twconvrsu_csv_v1 \
    --output_dir=datasets/convusersec/twconvrsu_tf_v1 \
    --max_sentence_len=1400

python3 -m datasets.embeddings_builder \
    datasets/convusersec/twconvrsu_tf_v1/vocabulary.txt \
    embeddings/fasttext/cc.es.300.vec



