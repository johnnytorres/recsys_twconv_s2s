
# exit on error
set -e

RUNNER=local

./scripts/experiments.sh 2011_trec random ${RUNNER}
./scripts/experiments.sh 2011_trec tfidf ${RUNNER}
./scripts/experiments.sh 2011_trec rnn ${RUNNER}
./scripts/experiments.sh 2011_trec lstm ${RUNNER}
./scripts/experiments.sh 2011_trec bilstm ${RUNNER}
./scripts/experiments.sh 2011_trec mf ${RUNNER}
./scripts/experiments.sh 2011_trec nmf ${RUNNER}
