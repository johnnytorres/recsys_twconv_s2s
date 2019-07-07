#!/usr/bin/env bash

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

# last argument can be
#--train
#--test
#--predict
#

#--embedding-path=~/data/ubuntu/bigtp_tf/embeddings.vec
#--predict-files=data/convusersec/twconvrsu_tf_activismexample.tfrecords