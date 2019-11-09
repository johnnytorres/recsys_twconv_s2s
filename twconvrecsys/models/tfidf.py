

import numpy as np

from twconvrecsys.data.csvreader import DataHandler
from twconvrecsys.metrics.recall import RecallEvaluator
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfPredictor:
	def __init__(self):
		self.vectorizer = TfidfVectorizer()

	def train(self, train_set):
		m = np.append(
			train_set.source.values,
			train_set.target.values)
		self.vectorizer.fit(m)

	def predict(self, source, targets):
		source_vec = self.vectorizer.transform([source])
		targets_vec = self.vectorizer.transform(targets)
		result = np.dot(targets_vec, source_vec.T)
		result = result.todense()
		result = np.asarray(result)
		result = result.flatten()
		result = np.argsort(result, axis=0)
		result = result[::-1]
		return result



