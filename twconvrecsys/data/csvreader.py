
import os
import pandas as pd


class DataHandler:
	@staticmethod
	def load_data(folder):
		folder = os.path.expanduser(folder)
		train = pd.read_csv(os.path.join(folder, 'train.csvrecords'))
		print(train.source.apply(lambda x: len(x.split(' '))).max())
		print(train.target.apply(lambda x: len(x.split(' '))).max())
		valid = pd.read_csv(os.path.join(folder, 'valid.csvrecords'))
		test = pd.read_csv(os.path.join(folder, 'test.csvrecords'))
		return train, valid, test

