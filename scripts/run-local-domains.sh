#!/usr/bin/env bash

python3 -m trainer.task
    --file-encoding=tf
    --train-files=data/convusersec/twconvrsu_tf_politics/train.tfrecords
    --train-batch-size=8
    --train-steps=1
    --num-epochs=1
    --eval-files=data/convusersec/twconvrsu_tf_politics/eval.tfrecords
    --eval-batch-size=10
    --eval-steps=1
    --test-files=data/convusersec/twconvrsu_tf_politics/test.tfrecords
    --predict-files=data/convusersec/twconvrsu_tf_politics/test.tfrecords
    --job-dir=data/convusersec/twconvrsu_tf_politics/models/
    --vocab-path=data/convusersec/twconvrsu_tf_politics/vocabulary.txt
    --vocab-proc=data/convusersec/twconvrsu_tf_politics/vocab_processor.bin
    --embedding-path=data/convusersec/twconvrsu_tf_politics/embeddings.vec
    --estimator=lstm
    --embedding-size=300
    --max-content-len=1400
    --max-utterance-len=1400
    --num-distractors=9
    --learning-rate=0.001
    --test

python3 -m trainer.task
    --file-encoding=tf
    --train-files=data/convusersec/twconvrsu_tf_sports/train.tfrecords
    --train-batch-size=8
    --train-steps=1
    --num-epochs=1
    --eval-files=data/convusersec/twconvrsu_tf_sports/eval.tfrecords
    --eval-batch-size=10
    --eval-steps=1
    --test-files=data/convusersec/twconvrsu_tf_sports/test.tfrecords
    --predict-files=data/convusersec/twconvrsu_tf_sports/test.tfrecords
    --job-dir=data/convusersec/twconvrsu_tf_sports/models/
    --vocab-path=data/convusersec/twconvrsu_tf_sports/vocabulary.txt
    --vocab-proc=data/convusersec/twconvrsu_tf_sports/vocab_processor.bin
    --embedding-path=data/convusersec/twconvrsu_tf_sports/embeddings.vec
    --estimator=lstm
    --embedding-size=300
    --max-content-len=1400
    --max-utterance-len=1400
    --num-distractors=9
    --learning-rate=0.001
    --test


python3 -m trainer.task
    --file-encoding=tf
    --train-files=data/convusersec/twconvrsu_tf_activism/train.tfrecords
    --train-batch-size=8
    --train-steps=1
    --num-epochs=1
    --eval-files=data/convusersec/twconvrsu_tf_activism/eval.tfrecords
    --eval-batch-size=10
    --eval-steps=1
    --test-files=data/convusersec/twconvrsu_tf_activism/test.tfrecords
    --predict-files=data/convusersec/twconvrsu_tf_activism/test.tfrecords
    --job-dir=data/convusersec/twconvrsu_tf_activism/models/
    --vocab-path=data/convusersec/twconvrsu_tf_activism/vocabulary.txt
    --vocab-proc=data/convusersec/twconvrsu_tf_activism/vocab_processor.bin
    --embedding-path=data/convusersec/twconvrsu_tf_activism/embeddings.vec
    --estimator=lstm
    --embedding-size=300
    --max-content-len=1400
    --max-utterance-len=1400
    --num-distractors=9
    --learning-rate=0.001
    --test