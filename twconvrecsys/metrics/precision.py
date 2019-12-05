
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score

class PrecisionEvaluator:
	@staticmethod
	def _calculate(y_true, y_pred, k=1):
		precision = 0
		for y_true_instance, y_pred_instance in zip(y_true, y_pred):
			y_true_instance = y_true_instance[:k]
			pos_label=1 if 1 in y_true_instance else 0
			#pos_label=1
			precision += precision_score(y_true_instance, y_pred_instance[:k], pos_label=pos_label)
		precision /= y_true.shape[0]
		return precision

	@staticmethod
	def calculate(y_true, y_pred):
		y_pred = np.array(y_pred) >= 0.5
		y_pred = y_pred.astype(int)


		enc = OneHotEncoder()
		y_true = np.array(y_true)
		y_true = y_true.reshape(-1, 1)
		y_true = enc.fit_transform(y_true)
		y_true = y_true.toarray()

		num_elements = y_pred.shape[1]
		klist = np.array( [1, 2, 5, 10])
		klist = klist[klist <= num_elements]
		metrics = []
		for k in klist:
			r = PrecisionEvaluator._calculate(y_true, y_pred, k)
			print('precision@({}, {}): {}'.format(k, num_elements, r))
			metrics.append(['recall', k, num_elements, r])
		return metrics


if __name__ == '__main__':
	labels = [1, 0, 2, 3]
	predictions = [
		[0.1, .52, .3, 0.8],
		[.1, .2, .3, 0.5],
		[.2, .3, .9, 0.],
		[.0, .1, .2, .7]
	]
	score = PrecisionEvaluator.calculate(labels, predictions)
	print (score)