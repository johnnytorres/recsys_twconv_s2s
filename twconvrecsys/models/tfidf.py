
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from twconvrecsys.models.random import RandomConversationRecommender


class TfidfPredictor(RandomConversationRecommender):
	def __init__(self):
		super().__init__()
		self.vectorizer = TfidfVectorizer()

	def train(self, train):
		source = train.source.values
		target = train.target.values
		m = np.append(source,target)
		self.vectorizer.fit(m)

	def _predict(self, source, targets):
		source_vec = self.vectorizer.transform([source])
		targets_vec = self.vectorizer.transform(targets.fillna(' '))
		result = np.dot(targets_vec, source_vec.T)
		result = result.todense()
		result = np.asarray(result)
		result = result.flatten()
		return result




