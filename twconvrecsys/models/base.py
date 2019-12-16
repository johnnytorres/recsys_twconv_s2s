
import numpy as np


class BaseConversationRecommender:
	def __init__(self):
		pass

	def train(self, train):
		pass

	def _predict(self, source, targets):
		pass

	def predict(self, test):
		predictions = []
		for ix, row in test.iterrows():
			y_pred = self._predict(row[0], row[1:-1])
			predictions.append(y_pred)
		predictions = np.array(predictions)
		return predictions


