from twconvrecusers.models.random import RandomConversationRecommender
from twconvrecusers.models.tfidf import TfidfPredictor


def get_model(args):
    if args.estimator == 'random':
        return RandomConversationRecommender()
    if args.estimator == 'tfidf':
        return TfidfPredictor()