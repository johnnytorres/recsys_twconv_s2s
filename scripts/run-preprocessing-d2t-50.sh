#!/usr/bin/env bash

python3 -m data.profiles_builder \
    --input=data/convusersec/timelines_raw.csv \
    --output=data/convusersec/timelines50.csv \
    --tokenize-tweets \
    --n=50

#conversations
# "dialogs" dataset is provided by crawler

#train
python3 -m data.csv_builder \
    --data-root=data/convusersec/dialogs \
    --profiles-path=data/convusersec/timelines50.csv \
    --output=data/convusersec/twconvrsu_csv_v2t_50/train.csv \
    --text-field=2 \
    --min-context-length=2 \
    --tokenize-tweets \
    -e=50000 \
    train

#valid
python3 -m data.csv_builder \
    --data-root=data/convusersec/dialogs \
    --profiles-path=data/convusersec/timelines50.csv \
    --output=data/convusersec/twconvrsu_csv_v2t_50/valid.csv \
    --text-field=2 \
    --min-context-length=2 \
    --tokenize-tweets \
    -e=5000 \
    valid

#tests
python3 -m data.csv_builder \
    --data-root=data/convusersec/dialogs \
    --profiles-path=data/convusersec/timelines50.csv \
    --output=data/convusersec/twconvrsu_csv_v2t_50/tests.csv \
    --text-field=2 \
    --min-context-length=2 \
    --tokenize-tweets \
    -e=5000 \
    tests

# tf records builder
python3 -m data.tfrecords_builder \
    --input_dir=data/convusersec/twconvrsu_csv_v2t_50 \
    --output_dir=data/convusersec/twconvrsu_tf_v2t_50 \
    --max_sentence_len=1400

python3 -m data.embeddings_builder \
    data/convusersec/twconvrsu_tf_v2t_50/vocabulary.txt \
    embeddings/fasttext/cc.es.300.vec



