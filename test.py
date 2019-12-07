import unittest
from train import MovieClassifierTrainer
from movie_classifier import MovieClassifier

class TestTrainModule(unittest.TestCase):

    def setUp(self):
        self.movietrainer = MovieClassifierTrainer()
        self.movietrainer.load_data()

    def test_rebalance(self):
        self.movietrainer.split_rebalance()
        self.movietrainer.prepare_features()

        self.assertEqual(self.movietrainer.train.shape[0],
                         self.movietrainer.X_train.shape[0],
                         "Train dimension should match")

        self.assertEqual(self.movietrainer.test.shape[0],
                         self.movietrainer.X_test.shape[0],
                         "Test dimension should match")

if __name__ == '__main__':
    unittest.main()
