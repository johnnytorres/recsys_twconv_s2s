

Code for implementation of the model in our paper:
https://doi.org/10.1016/j.eswa.2020.113270
 
To run the experiments, we recommend to create the python virtual environment and install prerequisites including Tensorflow.

experiments with sample data
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
make run DATASET=twconv_2011_trec_v10 SUBSET=alldataconvs MODEL=random  
make run DATASET=twconv_2011_trec_v10 SUBSET=alldataconvs MODEL=tfidf  
make run DATASET=twconv_2011_trec_v10 SUBSET=alldataconvs MODEL=rnn  
make run DATASET=twconv_2011_trec_v10 SUBSET=alldataconvs MODEL=lstm  
make run DATASET=twconv_2011_trec_v10 SUBSET=alldataconvs MODEL=bilstm  

make run DATASET=twconv_2011_trec_v10 SUBSET=alldatausers MODEL=random  
make run DATASET=twconv_2011_trec_v10 SUBSET=alldatausers MODEL=tfidf  
make run DATASET=twconv_2011_trec_v10 SUBSET=alldatausers MODEL=rnn  
make run DATASET=twconv_2011_trec_v10 SUBSET=alldatausers MODEL=lstm  
make run DATASET=twconv_2011_trec_v10 SUBSET=alldatausers MODEL=bilstm  

make run DATASET=twconv_2016_usersec_v10 SUBSET=alldataconvs MODEL=random  
make run DATASET=twconv_2016_usersec_v10 SUBSET=alldataconvs MODEL=tfidf  
make run DATASET=twconv_2016_usersec_v10 SUBSET=alldataconvs MODEL=rnn  
make run DATASET=twconv_2016_usersec_v10 SUBSET=alldataconvs MODEL=lstm  
make run DATASET=twconv_2016_usersec_v10 SUBSET=alldataconvs MODEL=bilstm  

make run DATASET=twconv_2016_usersec_v10 SUBSET=alldatausers MODEL=random  
make run DATASET=twconv_2016_usersec_v10 SUBSET=alldatausers MODEL=tfidf  
make run DATASET=twconv_2016_usersec_v10 SUBSET=alldatausers MODEL=rnn  
make run DATASET=twconv_2016_usersec_v10 SUBSET=alldatausers MODEL=lstm  
make run DATASET=twconv_2016_usersec_v10 SUBSET=alldatausers MODEL=bilstm  
```

experiments for embeddings analysis
```shell script
make run DATASET=twconv_2011_trec_v10 SUBSET=alldataconvs MODEL=rnn EMBEDDING_ENABLED=1 EMBEDDING_TRAINABLE=0   TAG="embeddings"
make run DATASET=twconv_2011_trec_v10 SUBSET=alldataconvs MODEL=rnn EMBEDDING_ENABLED=1 EMBEDDING_TRAINABLE=1   TAG="embeddings_trainable"
make run DATASET=twconv_2011_trec_v10 SUBSET=alldatausers MODEL=rnn EMBEDDING_ENABLED=1 EMBEDDING_TRAINABLE=0   TAG="embeddings"
make run DATASET=twconv_2011_trec_v10 SUBSET=alldatausers MODEL=rnn EMBEDDING_ENABLED=1 EMBEDDING_TRAINABLE=1   TAG="embeddings_trainable"

make run DATASET=twconv_2011_trec_v10 SUBSET=alldataconvs MODEL=lstm EMBEDDING_ENABLED=1 EMBEDDING_TRAINABLE=0   TAG="embeddings"
make run DATASET=twconv_2011_trec_v10 SUBSET=alldataconvs MODEL=lstm EMBEDDING_ENABLED=1 EMBEDDING_TRAINABLE=1   TAG="embeddings_trainable"
make run DATASET=twconv_2011_trec_v10 SUBSET=alldatausers MODEL=lstm EMBEDDING_ENABLED=1 EMBEDDING_TRAINABLE=0   TAG="embeddings"
make run DATASET=twconv_2011_trec_v10 SUBSET=alldatausers MODEL=lstm EMBEDDING_ENABLED=1 EMBEDDING_TRAINABLE=1   TAG="embeddings_trainable"


make run DATASET=twconv_2016_usersec_v10 SUBSET=alldataconvs MODEL=rnn EMBEDDING_ENABLED=1 EMBEDDING_TRAINABLE=0   TAG="embeddings"
make run DATASET=twconv_2016_usersec_v10 SUBSET=alldataconvs MODEL=rnn EMBEDDING_ENABLED=1 EMBEDDING_TRAINABLE=1   TAG="embeddings_trainable"
make run DATASET=twconv_2016_usersec_v10 SUBSET=alldatausers MODEL=rnn EMBEDDING_ENABLED=1 EMBEDDING_TRAINABLE=0   TAG="embeddings"
make run DATASET=twconv_2016_usersec_v10 SUBSET=alldatausers MODEL=rnn EMBEDDING_ENABLED=1 EMBEDDING_TRAINABLE=1   TAG="embeddings_trainable"
```

experiments for different training size
```shell script
make run DATASET=twconv_2011_trec_v10 SUBSET=alldataconvs_test20 MODEL=rnn  
make run DATASET=twconv_2011_trec_v10 SUBSET=alldataconvs_test30 MODEL=rnn  
make run DATASET=twconv_2011_trec_v10 SUBSET=alldataconvs_test40 MODEL=rnn  

make run DATASET=twconv_2016_usersec_v10 SUBSET=alldataconvs_test20 MODEL=rnn  
make run DATASET=twconv_2016_usersec_v10 SUBSET=alldataconvs_test30 MODEL=rnn  
make run DATASET=twconv_2016_usersec_v10 SUBSET=alldataconvs_test40 MODEL=rnn  
```

calculate metrics
```shell script
./scripts/results.sh
```