#!/usr/bin/env bash

python3 -m dataset.profiles_builder \
    --input=dataset/convusersec/timelines_raw.csv \
    --output=dataset/convusersec/timelines50.csv \
    --tokenize-tweets \
    --n=50

#conversations
# "dialogs" dataset is provided by crawler

#train
python3 -m dataset.csv_builder \
    --dataset-root=dataset/convusersec/dialogs \
    --profiles-path=dataset/convusersec/timelines50.csv \
    --output=dataset/convusersec/twconvrsu_csv_v2t_50/train.csv \
    --text-field=2 \
    --min-context-length=2 \
    --tokenize-tweets \
    -e=50000 \
    train

#valid
python3 -m dataset.csv_builder \
    --dataset-root=dataset/convusersec/dialogs \
    --profiles-path=dataset/convusersec/timelines50.csv \
    --output=dataset/convusersec/twconvrsu_csv_v2t_50/valid.csv \
    --text-field=2 \
    --min-context-length=2 \
    --tokenize-tweets \
    -e=5000 \
    valid

#tests
python3 -m dataset.csv_builder \
    --dataset-root=dataset/convusersec/dialogs \
    --profiles-path=dataset/convusersec/timelines50.csv \
    --output=dataset/convusersec/twconvrsu_csv_v2t_50/tests.csv \
    --text-field=2 \
    --min-context-length=2 \
    --tokenize-tweets \
    -e=5000 \
    tests

# tf records builder
python3 -m dataset.tfrecords_builder \
    --input_dir=dataset/convusersec/twconvrsu_csv_v2t_50 \
    --output_dir=dataset/convusersec/twconvrsu_tf_v2t_50 \
    --max_sentence_len=1400

python3 -m dataset.embeddings_builder \
    dataset/convusersec/twconvrsu_tf_v2t_50/vocabulary.txt \
    embeddings/fasttext/cc.es.300.vec



