

import numpy as np

from data_handler import DataHandler
from evaluation_handler import EvaluationHandler
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfPredictor:
	def __init__(self):
		self.vectorizer = TfidfVectorizer()

	def train(self, training_set):
		m = np.append(
			training_set.Context.values,
			training_set.Utterance.values)
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


if __name__=='__main__':
	data_handler = DataHandler()
	predictor = TfidfPredictor()
	train, valid, test = data_handler.load_data('~/data/ubuntu')
	predictor.train(train)
	y_pred = [predictor.predict(row['Context'], row[1:]) for ix, row in test.iterrows()]
	y_true = np.zeros(test.shape[0])
	EvaluationHandler.evaluate_predictor(y_true, y_pred)
