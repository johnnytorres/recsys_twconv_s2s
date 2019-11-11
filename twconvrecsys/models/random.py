
import numpy as np

from twconvrecsys.models.base import BaseConversationRecommender


class RandomConversationRecommender(BaseConversationRecommender):
	def __init__(self):
		super().__init__()

	def _predict(self, source, targets):
		n_utt = len(targets)
		return np.random.choice(n_utt, n_utt, replace=False)

