
# exit on error
set -e

RUNNER=local

./scripts/experiments.sh ${RUNNER} 2011_trec random
./scripts/experiments.sh ${RUNNER} 2011_trec tfidf
./scripts/experiments.sh ${RUNNER} 2011_trec rnn
./scripts/experiments.sh ${RUNNER} 2011_trec lstm
./scripts/experiments.sh ${RUNNER} 2011_trec bilstm
./scripts/experiments.sh ${RUNNER} 2011_trec mf
./scripts/experiments.sh ${RUNNER} 2011_trec nmf
