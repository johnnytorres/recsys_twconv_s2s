
class EvaluationHandler:
	@staticmethod
	def evaluate_recall(y_true, y_pred, k=1):
		num_examples = len(y_true)
		num_correct = 0
		for label, predictions in zip(y_true, y_pred):
			if label in predictions[:k]:
				num_correct+=1
		return num_correct/num_examples

	@staticmethod
	def evaluate_predictor(y_true, y_pred):
		for k in [1, 2, 5, 10]:
			r = EvaluationHandler.evaluate_recall(y_true, y_pred, k)
			print(f'recall@({k}, 10): {r}')


if __name__=='__main__':
	labels = [1, 0, 2, 4]
	predictions = [
		[1, 2, 3, 0],
		[1, 2, 3, 0],
		[2, 3, 1, 0],
		[0, 1, 2, 3]
	]
	score = EvaluationHandler.evaluate_recall(labels, predictions)
	print (score)