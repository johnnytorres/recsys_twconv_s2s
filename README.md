
# prepare the datasets


# prepare the python virtual environment

create the python virtual environment

install prerequisites

enable jupyter inside the virtual environment
https://janakiev.com/blog/jupyter-virtual-envs/



this will run the experiments 
```shell script
./scripts/run.sh 
```

results 
tfidf
recall@(1, 6): 0.42857142857142855
recall@(2, 6): 0.5714285714285714
recall@(5, 6): 1.0
[['recall', 1, 6, 0.42857142857142855], ['recall', 2, 6, 0.5714285714285714], ['recall', 5, 6, 1.0]]


rnn
recall_at_1 = 0.42857142857142855, recall_at_2 = 0.5714285714285714, recall_at_5 = 1.0
[['recall', 1, 6, 0.42857142857142855], ['recall', 2, 6, 0.5714285714285714], ['recall', 5, 6, 1.0]]



sample predictions (sample data, set set), the last-1 column is the label, last column indicates if it's in top k=2

tfidf r@k
[0, 3, 5, 1, 4, 2, 5],
[4, 5, 3, 0, 2, 1, 4],1
[2, 3, 5, 0, 4, 1, 2],1
[0, 2, 5, 3, 1, 4, 3],
[1, 3, 2, 4, 0, 5, 1],1
[1, 0, 3, 5, 2, 4, 1],1
[4, 5, 0, 1, 2, 3, 1],
[1, 4, 3, 0, 5, 2, 4],1
[2, 4, 1, 0, 5, 3, 1],
[0, 1, 4, 5, 2, 3, 1],1
[2, 1, 5, 4, 0, 3, 4],
[2, 1, 5, 3, 0, 4, 2],1
[5, 0, 4, 2, 1, 3, 2],
[4, 0, 5, 2, 1, 3, 4],1

rnn r@k
[2, 0, 5, 4, 3, 1, 5],
[5, 1, 4, 3, 2, 0, 4],
[2, 5, 4, 3, 1, 0, 2],1
[3, 5, 4, 2, 1, 0, 3],1
[1, 5, 4, 3, 2, 0, 1],1
[4, 1, 5, 3, 2, 0, 1],1
[5, 4, 3, 2, 1, 0, 1],
[4, 5, 3, 2, 1, 0, 4],1
[5, 4, 3, 2, 1, 0, 1],
[1, 0, 4, 5, 3, 2, 1],1
[0, 5, 4, 3, 2, 1, 4],
[2, 1, 0, 5, 4, 3, 2],1
[3, 4, 2, 5, 1, 0, 2],
[5, 4, 3, 2, 1, 0, 4],1


for predictions

{'source': <tf.Tensor 'IteratorGetNext:0' shape=(?, 10) dtype=int64>, 'source_len': <tf.Tensor 'IteratorGetNext:1' shape=(?, 1) dtype=int64>, 'target': <tf.Tensor 'IteratorGetNext:2' shape=(?, 10) dtype=int64>, 'target_len': <tf.Tensor 'IteratorGetNext:3' shape=(?, 1) dtype=int64>}


sample data experiments
```shell script
make run \
    DATASET=twconv_2011_trec_v10 \
    SUBSET=sampledataconvs \
    MODEL=rnn \
    TRAIN_BATCH_SIZE=8 \
    NUM_EPOCHS=2 \
    MAX_INPUT_LEN=10 \
    MAX_SOURCE_LEN=10 \
    MAX_TARGET_LEN=10 \
    NUM_DISTRACTORS=5 \
    EMBEDDING_SIZE=300 \
    EMBEDDING_ENABLED=1 \
    EMBEDDING_TRAINABLE=1 \
    RNN_DIM=50
```

experiments for benchmarking models
```shell script
make run DATASET=twconv_2011_trec_v10 SUBSET=alldataconvs MODEL=random RUN=gcloud STORE=gcloud
make run DATASET=twconv_2011_trec_v10 SUBSET=alldataconvs MODEL=tfidf RUN=gcloud STORE=gcloud
make run DATASET=twconv_2011_trec_v10 SUBSET=alldataconvs MODEL=rnn RUN=gcloud STORE=gcloud
make run DATASET=twconv_2011_trec_v10 SUBSET=alldataconvs MODEL=lstm RUN=gcloud STORE=gcloud
make run DATASET=twconv_2011_trec_v10 SUBSET=alldataconvs MODEL=bilstm RUN=gcloud STORE=gcloud

make run DATASET=twconv_2011_trec_v10 SUBSET=alldatausers MODEL=random RUN=gcloud STORE=gcloud
make run DATASET=twconv_2011_trec_v10 SUBSET=alldatausers MODEL=tfidf RUN=gcloud STORE=gcloud
make run DATASET=twconv_2011_trec_v10 SUBSET=alldatausers MODEL=rnn RUN=gcloud STORE=gcloud
make run DATASET=twconv_2011_trec_v10 SUBSET=alldatausers MODEL=lstm RUN=gcloud STORE=gcloud
make run DATASET=twconv_2011_trec_v10 SUBSET=alldatausers MODEL=bilstm RUN=gcloud STORE=gcloud

make run DATASET=twconv_2016_usersec_v10 SUBSET=alldataconvs MODEL=random RUN=gcloud STORE=gcloud
make run DATASET=twconv_2016_usersec_v10 SUBSET=alldataconvs MODEL=tfidf RUN=gcloud STORE=gcloud
make run DATASET=twconv_2016_usersec_v10 SUBSET=alldataconvs MODEL=rnn RUN=gcloud STORE=gcloud
make run DATASET=twconv_2016_usersec_v10 SUBSET=alldataconvs MODEL=lstm RUN=gcloud STORE=gcloud
make run DATASET=twconv_2016_usersec_v10 SUBSET=alldataconvs MODEL=bilstm RUN=gcloud STORE=gcloud

make run DATASET=twconv_2016_usersec_v10 SUBSET=alldatausers MODEL=random RUN=gcloud STORE=gcloud
make run DATASET=twconv_2016_usersec_v10 SUBSET=alldatausers MODEL=tfidf RUN=gcloud STORE=gcloud
make run DATASET=twconv_2016_usersec_v10 SUBSET=alldatausers MODEL=rnn RUN=gcloud STORE=gcloud
make run DATASET=twconv_2016_usersec_v10 SUBSET=alldatausers MODEL=lstm RUN=gcloud STORE=gcloud
make run DATASET=twconv_2016_usersec_v10 SUBSET=alldatausers MODEL=bilstm RUN=gcloud STORE=gcloud
```

experiments for embeddings
```shell script
make run DATASET=twconv_2011_trec_v10 SUBSET=alldataconvs MODEL=rnn EMBEDDING_ENABLED=1 EMBEDDING_TRAINABLE=0 RUN=gcloud STORE=gcloud TAG="embeddings"
make run DATASET=twconv_2011_trec_v10 SUBSET=alldataconvs MODEL=rnn EMBEDDING_ENABLED=1 EMBEDDING_TRAINABLE=1 RUN=gcloud STORE=gcloud TAG="embeddings_trainable"
make run DATASET=twconv_2011_trec_v10 SUBSET=alldatausers MODEL=rnn EMBEDDING_ENABLED=1 EMBEDDING_TRAINABLE=0 RUN=gcloud STORE=gcloud TAG="embeddings"
make run DATASET=twconv_2011_trec_v10 SUBSET=alldatausers MODEL=rnn EMBEDDING_ENABLED=1 EMBEDDING_TRAINABLE=1 RUN=gcloud STORE=gcloud TAG="embeddings_trainable"

make run DATASET=twconv_2011_trec_v10 SUBSET=alldataconvs MODEL=lstm EMBEDDING_ENABLED=1 EMBEDDING_TRAINABLE=0 RUN=gcloud STORE=gcloud TAG="embeddings"
make run DATASET=twconv_2011_trec_v10 SUBSET=alldataconvs MODEL=lstm EMBEDDING_ENABLED=1 EMBEDDING_TRAINABLE=1 RUN=gcloud STORE=gcloud TAG="embeddings_trainable"
make run DATASET=twconv_2011_trec_v10 SUBSET=alldatausers MODEL=lstm EMBEDDING_ENABLED=1 EMBEDDING_TRAINABLE=0 RUN=gcloud STORE=gcloud TAG="embeddings"
make run DATASET=twconv_2011_trec_v10 SUBSET=alldatausers MODEL=lstm EMBEDDING_ENABLED=1 EMBEDDING_TRAINABLE=1 RUN=gcloud STORE=gcloud TAG="embeddings_trainable"


make run DATASET=twconv_2016_usersec_v10 SUBSET=alldataconvs MODEL=rnn EMBEDDING_ENABLED=1 EMBEDDING_TRAINABLE=0 RUN=gcloud STORE=gcloud TAG="embeddings"
make run DATASET=twconv_2016_usersec_v10 SUBSET=alldataconvs MODEL=rnn EMBEDDING_ENABLED=1 EMBEDDING_TRAINABLE=1 RUN=gcloud STORE=gcloud TAG="embeddings_trainable"
make run DATASET=twconv_2016_usersec_v10 SUBSET=alldatausers MODEL=rnn EMBEDDING_ENABLED=1 EMBEDDING_TRAINABLE=0 RUN=gcloud STORE=gcloud TAG="embeddings"
make run DATASET=twconv_2016_usersec_v10 SUBSET=alldatausers MODEL=rnn EMBEDDING_ENABLED=1 EMBEDDING_TRAINABLE=1 RUN=gcloud STORE=gcloud TAG="embeddings_trainable"
```

experiments for train size
```shell script
make run DATASET=twconv_2011_trec_v10 SUBSET=alldataconvs_test20 MODEL=rnn RUN=gcloud STORE=gcloud
make run DATASET=twconv_2011_trec_v10 SUBSET=alldataconvs_test30 MODEL=rnn RUN=gcloud STORE=gcloud
make run DATASET=twconv_2011_trec_v10 SUBSET=alldataconvs_test40 MODEL=rnn RUN=gcloud STORE=gcloud

make run DATASET=twconv_2016_usersec_v10 SUBSET=alldataconvs_test20 MODEL=rnn RUN=gcloud STORE=gcloud
make run DATASET=twconv_2016_usersec_v10 SUBSET=alldataconvs_test30 MODEL=rnn RUN=gcloud STORE=gcloud
make run DATASET=twconv_2016_usersec_v10 SUBSET=alldataconvs_test40 MODEL=rnn RUN=gcloud STORE=gcloud
```

collect results and calculate metrics
```shell script
./scripts/results.sh
```