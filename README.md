
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


experiments
```shell script
./scripts/experiments-rnn.sh twconv_2011_trec_v3 alldata 24592
```