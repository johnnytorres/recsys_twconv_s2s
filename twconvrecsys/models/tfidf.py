
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from twconvrecsys.models.random import RandomConversationRecommender


class TfidfPredictor(RandomConversationRecommender):
	def __init__(self):
		super().__init__()
		self.vectorizer = TfidfVectorizer()

	def train(self, train):
		m = np.append(
			train.source.values,
			train.target.values)
		self.vectorizer.fit(m)

	def _predict(self, source, targets):
		source_vec = self.vectorizer.transform([source])
		targets_vec = self.vectorizer.transform(targets.fillna(' '))
		result = np.dot(targets_vec, source_vec.T)
		result = result.todense()
		result = np.asarray(result)
		result = result.flatten()
		# result = np.argsort(result, axis=0)
		# result = result[::-1]
		return result

	# def predict(self, test):
	# 	y_pred = [self._predict(row[0], row[1:-1]) for ix, row in test.iterrows()]
	# 	y_pred = np.array(y_pred)
	# 	return y_pred




