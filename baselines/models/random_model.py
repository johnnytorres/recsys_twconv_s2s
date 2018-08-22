
import numpy as np

from data_handler import DataHandler
from evaluation_handler import EvaluationHandler


class RandomPredictor:
	def __init__(self):
		pass

	def predict(self, context, utterances):
		n_utt = len(utterances)
		return np.random.choice(n_utt, n_utt, replace=False)


if __name__=='__main__':
	data_handler = DataHandler()
	predictor = RandomPredictor()
	train, valid, test = data_handler.load_data('~/data/ubuntu')
	y_pred = [predictor.predict(row['Context'], row[1:]) for ix, row in test.iterrows()]
	y_true = np.zeros(test.shape[0])
	EvaluationHandler.evaluate_predictor(y_true, y_pred)
