
import os
import pandas as pd


class DataHandler:
	@staticmethod
	def load_data(args):
		folder = os.path.join(args.data_dir, args.data_subdir)
		train = pd.read_csv(os.path.join(folder, 'train.csvrecords'))
		print(train.source.apply(lambda x: len(x.split(' '))).max())
		print(train.target.apply(lambda x: len(x.split(' '))).max())
		valid = pd.read_csv(os.path.join(folder, 'valid.csvrecords'))
		test = pd.read_csv(os.path.join(folder, 'test.csvrecords'))
		return train, valid, test

	@staticmethod
	def load_test_data(args):
		folder = os.path.join(args.data_dir, args.data_subdir)
		test = pd.read_csv(os.path.join(folder, 'test.csvrecords'))
		return test
