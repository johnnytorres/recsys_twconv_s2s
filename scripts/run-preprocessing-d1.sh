#!/usr/bin/env bash

#profiles

python3 -m dataset.profiles_builder \
    --input=dataset/convusersec/timelines_raw.csv \
    --output=dataset/convusersec/timelines.csv \
    --n=10


#conversations

# trainset
python3 -m dataset.csv_builder \
    --dataset-root=dataset/convusersec/dialogs \
    --profiles-path=dataset/convusersec/timelines.csv \
    --output=dataset/convusersec/twconvrsu_csv_v1/train.csv \
    --text-field=2 \
    --min-context-length=2 \
    --tokenize-punk \
    -e=50000 \
    train

# valid set
python3 -m dataset.csv_builder \
    --dataset-root=dataset/convusersec/dialogs \
    --profiles-path=dataset/convusersec/timelines.csv \
    --output=dataset/convusersec/twconvrsu_csv_v1/valid.csv \
    --text-field=2 \
    --min-context-length=2 \
    --tokenize-punk \
    -e=5000 \
    valid

#tests
python3 -m dataset.csv_builder \
    --dataset-root=dataset/convusersec/dialogs \
    --profiles-path=dataset/convusersec/timelines.csv \
    --output=dataset/convusersec/twconvrsu_csv_v1/tests.csv \
    --text-field=2 \
    --min-context-length=2 \
    --tokenize-punk \
    -e=5000 \
    tests

# tf records builder
python3 -m dataset.tfrecords_builder \
    --input_dir=dataset/convusersec/twconvrsu_csv_v1 \
    --output_dir=dataset/convusersec/twconvrsu_tf_v1 \
    --max_sentence_len=1400

python3 -m dataset.embeddings_builder \
    dataset/convusersec/twconvrsu_tf_v1/vocabulary.txt \
    embeddings/fasttext/cc.es.300.vec



