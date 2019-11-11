
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

tfidf
recall@(1, 6): 0.42857142857142855
recall@(2, 6): 0.5714285714285714
recall@(5, 6): 1.0

lstm
accuracy = 0.6547619, accuracy_baseline = 0.8333333, auc = 0.6183674, auc_precision_recall = 0.39766663, average_loss = 3.7728863, global_step = 74, label/mean = 0.16666667, loss = 3.7728863, precision = 0.2413793, prediction/mean = 0.34226888, recall = 0.5, recall_at_1 = 0.42857142857142855, recall_at_2 = 0.6428571428571429, recall_at_5 = 0.7857142857142857

mf
accuracy = 0.5833333, accuracy_baseline = 0.8333333, auc = 0.7750001, auc_precision_recall = 0.6067064, average_loss = 6.403243, global_step = 74, label/mean = 0.16666667, loss = 6.403243, precision = 0.27659574, prediction/mean = 0.558093, recall = 0.9285714, recall_at_1 = 0.5714285714285714, recall_at_2 = 0.7857142857142857, recall_at_5 = 1.0


 