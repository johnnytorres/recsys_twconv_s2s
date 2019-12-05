import unittest
import numpy as np
from twconvrecsys.data.csvreader import DataHandler
from twconvrecsys.metrics.recall import RecallEvaluator
from twconvrecsys.models.tfidf import TfidfPredictor


class TestCaseTfidf(unittest.TestCase):
    def test_tfidf(self):
        data_dir = '~/data/twconv/2011_trec/sampledata'
        #parser = argparse.Parse
        data_handler = DataHandler()
        predictor = TfidfPredictor()
        train, valid, test = data_handler.load_data(data_dir)
        predictor.train(train)
        y_pred = predictor.predict(test)
        y_true = test.label.values
        metrics = RecallEvaluator.calculate(y_true, y_pred)
        self.assertAlmostEqual(metrics[0][3], 0.42857142857142855)
        self.assertAlmostEqual(metrics[1][3], 0.5714285714285714)



if __name__ == '__main__':
    unittest.main()
