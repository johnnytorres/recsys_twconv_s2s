
import os
import pandas as pd


class DataHandler:
	@staticmethod
	def load_data(folder):
		folder = os.path.expanduser(folder)
		train = pd.read_csv(os.path.join(folder, 'train.csv'))
		valid = pd.read_csv(os.path.join(folder, 'valid.csv'))
		test = pd.read_csv(os.path.join(folder, 'test.csv'))
		return train, valid, test

