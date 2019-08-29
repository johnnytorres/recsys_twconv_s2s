#!/usr/bin/env bash

python3 -m datasets.profiles_builder \
    --input=datasets/convusersec/timelines_raw.csv \
    --output=datasets/convusersec/timelines50.csv \
    --tokenize-tweets \
    --n=50

#conversations
# "dialogs" datasets is provided by crawler

#train
python3 -m datasets.csv_builder \
    --datasets-root=datasets/convusersec/dialogs \
    --profiles-path=datasets/convusersec/timelines50.csv \
    --output=datasets/convusersec/twconvrsu_csv_v2t_50/train.csv \
    --text-field=2 \
    --min-context-length=2 \
    --tokenize-tweets \
    -e=50000 \
    train

#valid
python3 -m datasets.csv_builder \
    --datasets-root=datasets/convusersec/dialogs \
    --profiles-path=datasets/convusersec/timelines50.csv \
    --output=datasets/convusersec/twconvrsu_csv_v2t_50/valid.csv \
    --text-field=2 \
    --min-context-length=2 \
    --tokenize-tweets \
    -e=5000 \
    valid

#tests
python3 -m datasets.csv_builder \
    --datasets-root=datasets/convusersec/dialogs \
    --profiles-path=datasets/convusersec/timelines50.csv \
    --output=datasets/convusersec/twconvrsu_csv_v2t_50/tests.csv \
    --text-field=2 \
    --min-context-length=2 \
    --tokenize-tweets \
    -e=5000 \
    tests

# tf records builder
python3 -m datasets.tfrecords_builder \
    --input_dir=datasets/convusersec/twconvrsu_csv_v2t_50 \
    --output_dir=datasets/convusersec/twconvrsu_tf_v2t_50 \
    --max_sentence_len=1400

python3 -m datasets.embeddings_builder \
    datasets/convusersec/twconvrsu_tf_v2t_50/vocabulary.txt \
    embeddings/fasttext/cc.es.300.vec



