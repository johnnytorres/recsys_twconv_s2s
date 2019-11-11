
import numpy as np


class RecallEvaluator:
	@staticmethod
	def calculate_recall(y_true, y_pred, k=1):
		num_examples = len(y_true)
		num_correct = 0
		for label, predictions in zip(y_true, y_pred):
			if label in predictions[:k]:
				num_correct+=1
		return num_correct/num_examples

	@staticmethod
	def evaluate(y_true, y_pred):
		y_pred = np.argsort(y_pred, axis=1)
		y_pred = np.fliplr(y_pred)
		num_elements = y_pred.shape[1]
		klist = np.array( [1, 2, 5, 10])
		klist = klist[klist < num_elements]
		metrics = []
		for k in klist:
			r = RecallEvaluator.calculate_recall(y_true, y_pred, k)
			print('recall@({}, {}): {}'.format(k, num_elements, r))
			metrics.append(['recall', k, num_elements, r])
		return metrics


if __name__ == '__main__':
	labels = [1, 0, 2, 4]
	predictions = [
		[1, 2, 3, 0],
		[1, 2, 3, 0],
		[2, 3, 1, 0],
		[0, 1, 2, 3]
	]
	score = RecallEvaluator.calculate_recall(labels, predictions)
	print (score)