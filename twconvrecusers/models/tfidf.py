

import numpy as np

from twconvrecusers.dataset.csvreader import DataHandler
from twconvrecusers.metrics.recall import RecallEvaluator
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfPredictor:
	def __init__(self):
		self.vectorizer = TfidfVectorizer()

	def train(self, train_set):
		m = np.append(
			train_set.context.values,
			train_set.profile.values)
		self.vectorizer.fit(m)

	def predict(self, context, utterances):
		context_vec = self.vectorizer.transform([context])
		utterances_vec = self.vectorizer.transform(utterances)
		result = np.dot(utterances_vec, context_vec.T)
		result = result.todense()
		result = np.asarray(result)
		result = result.flatten()
		result = np.argsort(result, axis=0)
		result = result[::-1]
		return result



