#!/usr/bin/env bash

python3 -m trainer.task
--file-encoding=tf
--train-files=dataset/convusersec/twconvrsu_tf_activism/train.tfrecords
--eval-files=dataset/convusersec/twconvrsu_tf_activism/eval.tfrecords
--tests-files=dataset/convusersec/twconvrsu_tf_activism/tests.tfrecords
--vocab-path=dataset/convusersec/twconvrsu_tf_activism/vocabulary.txt
--vocab-proc=dataset/convusersec/twconvrsu_tf_activism/vocab_processor.bin
--embedding-path=dataset/convusersec/twconvrsu_tf_activism/embeddings.vec
--train-batch-size=8
--train-steps=1
--num-epochs=1
--eval-batch-size=10
--eval-steps=1
--estimator=lstm
--embedding-size=300
--max-content-len=1400
--max-utterance-len=1400
--num-distractors=9
--learning-rate=0.001

# last argument can be
#--train
#--tests
#--predict
#

#--embedding-path=~/dataset/ubuntu/bigtp_tf/embeddings.vec
#--predict-files=dataset/convusersec/twconvrsu_tf_activismexample.tfrecords


--job-dir=~/dataset/twconv/trec/resultstagging/model_uniforminit_noreg_50it
--train-files=~/dataset/twconv/trec/datastagging/train.tfrecords
--eval-files=~/dataset/twconv/trec/datastagging/valid.tfrecords
--test-files=~/dataset/twconv/trec/datastagging/test.tfrecords
--vocab-path=~/dataset/twconv/trec/datastagging/vocabulary.txt
--file-encoding=tf
--test
--num-distractor=5
--max-content-len=10
--max-utterance-len=10
--train-steps=50
--train-batch-size=2
--num-epochs=1
--eval-every-secs=1
--eval-batch-size=10
--eval-steps=1

python -m twconvrecusers.task \
  --dataset-dir=~/dataset/twconv/trec/datastagging \
  --job-dir=~/dataset/twconv/trec/resultstagging/rnn \
  lstm \
  --train \
  --num-distractor=5 \
  --max-content-len=10 \
  --max-utterance-len=10 \
  --train-steps=50 \
  --train-batch-size=2 \
  --num-epochs=1 \
  --eval-every-secs=1 \
  --eval-batch-size=10 \
  --eval-steps=1