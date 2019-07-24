
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

	@staticmethod
	def show_training_set_stats(training_set):
		import matplotlib.pyplot as plt
		import matplotlib
		matplotlib.style.use('ggplot')
		print(training_set.describe())
		training_set.Label.hist()
		plt.show()
		plt.waitforbuttonpress()


if __name__=='__main__':
	train, valid, test = DataHandler.load_data('~/data/ubuntu')
	DataHandler.show_training_set_stats(train)