import unittest
from train import MovieClassifierTrainer, StemmedTfidfVectorizer, stop_words, stemmer
from movie_classifier import MovieClassifier


# test should be run after dataset definition
class TestTrainModule(unittest.TestCase):

    def setUp(self):
        self.movietrainer = MovieClassifierTrainer()
        self.movietrainer.load_data()

    def test_train_size(self):
        self.movietrainer.split_rebalance()
        self.movietrainer.prepare_features()
        self.assertEqual(self.movietrainer.train.shape[0],
                         self.movietrainer.X_train.shape[0],
                         "Train dimension should match")
        self.assertEqual(self.movietrainer.test.shape[0],
                         self.movietrainer.X_test.shape[0],
                         "Test dimension should match")



class TestMovieClassifier(unittest.TestCase):

    def setUp(self):
        self.mclf = MovieClassifier()

    def test_model_load(self):
        self.mclf.load_models('models/')
        self.assertTrue(self.mclf.movieclf, "Model not loaded")

    def test_svd_load(self):
        self.mclf.load_models('models/')
        self.assertTrue(self.mclf.svd, "SVD object not loaded")

    def test_tfidf_load(self):
        self.mclf.load_models('models/')
        self.assertTrue(self.mclf.tfidf, "TFIDF object not loaded")

    def test_output(self):
        self.mclf.load_models('models/')
        result = self.mclf.get_genre('Some Title', 'Some Description')

        self.assertTrue('title' in result.keys(), 'Title missing in output')
        self.assertTrue('description' in result.keys(), 'Description missing in output')
        self.assertTrue('genre' in result.keys(), 'Genre missing in output')

if __name__ == '__main__':
    unittest.main()
