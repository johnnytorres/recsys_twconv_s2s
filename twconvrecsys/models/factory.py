from twconvrecsys.models.random import RandomConversationRecommender
from twconvrecsys.models.tfidf import TfidfPredictor


class ModelName:
    RANDOM='random'
    TFIDF='tfidf'
    RNN='rnn'
    LSTM='lstm'
    BILSTM='bilstm'
    MF='mf'
    NMF='nmf'



def get_model(args):
    if args.estimator == 'random':
        return RandomConversationRecommender()
    if args.estimator == 'tfidf':
        return TfidfPredictor()