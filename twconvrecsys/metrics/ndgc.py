
import numpy as np
import math
from sklearn.preprocessing import OneHotEncoder


class NdgcEvaluator:
	@staticmethod
	def get_ndgc(ranklist, label):
		for i in range(len(ranklist)):
			item = ranklist[i]
			if item == label:
				return math.log(2) / math.log(i + 2)
		return 0

	@staticmethod
	def _calculate(y_true, y_pred, k=1):
		num_examples = len(y_true)
		ndgc = 0
		for label, predictions in zip(y_true, y_pred):
			ranklist = predictions[:k]
			ndgc += NdgcEvaluator.get_ndgc(ranklist, label)
		return ndgc / num_examples

	@staticmethod
	def calculate(y_true, y_pred):
		y_pred = np.argsort(y_pred, axis=1)
		y_pred = np.fliplr(y_pred)
		num_elements = y_pred.shape[1]
		klist = np.array([1, 2, 5, 10])
		klist = klist[klist <= num_elements]
		metrics = []
		for k in klist:
			r = NdgcEvaluator._calculate(y_true, y_pred, k)
			print('ndgc@({}, {}): {}'.format(k, num_elements, r))
			metrics.append(['ndgc', k, num_elements, r])
		return metrics


if __name__ == '__main__':
	labels = [1, 0, 2, 3]
	predictions = [
		[0.1, .52, .3, 0.8],
		[.1, .2, .3, 0.5],
		[.2, .3, .9, 0.],
		[.0, .1, .2, .7]
	]
	score = NdgcEvaluator.calculate(labels, predictions)
	print (score)