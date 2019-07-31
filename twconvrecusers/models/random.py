
import numpy as np


class RandomConversationRecommender:
	def __init__(self):
		pass

	def predict(self, context, utterances):
		n_utt = len(utterances)
		return np.random.choice(n_utt, n_utt, replace=False)


