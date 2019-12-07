import pandas as pd
import json
import ast
import numpy as np

import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

from joblib import dump

import argparse

stop_words = nltk.corpus.stopwords.words('english')
stemmer = nltk.stem.snowball.EnglishStemmer()

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (stemmer.stem(w) for w in analyzer(doc))


class MovieClassifierTrainer:


    def extract_data(self, source):
        movies = pd.read_csv(source)

        # transform from string to objects
        movies['genres'] = movies['genres'].apply(lambda x: np.nan if pd.isnull(x) else ast.literal_eval(x))

        # extract main genres list
        genres = []
        for m in movies['genres']:
            if m != []:
                genres.append(m[0])

        # filter low frequency genres to ease classifier training
        genre_list = pd.DataFrame(genres)['name'].value_counts()
        genre_list = genre_list[genre_list.values > 500]
        main_genres = set(genre_list.keys().tolist())

        # define dataset for classifier
        target = movies[['id', 'title', 'overview', 'genres']]

        # remove movies without overview or title
        target = target[(~target['overview'].isnull()) & (~target['title'].isnull())]

        # extract only the first genre to ease the classifier training
        target['genre_list'] = target['genres'].apply(lambda x: self.__get_genres_list(x))
        target['genre'] = target['genre_list'].apply(lambda x: self.__get_genre_in_top(x, main_genres))
        target = target.drop(['genres', 'genre_list'], axis=1)

        # remove movies that do not have genre or do not belong to main list
        target = target[~target['genre'] == 'no main genre']
        target.to_csv('data.csv', index=None)

        self.data = target

    def prepare_features(self, svd_dim=600):

        # preprocessing: define input text as concatenation of title and overview
        self.data['text'] = self.data.apply(lambda x: x['title'] + ' ' + x['overview'], axis=1)

        # split train and test data
        train, test = train_test_split(self.data, test_size=0.1, random_state=17)

        tfidf = StemmedTfidfVectorizer(stop_words=stop_words)
        svd = TruncatedSVD(n_components=svd_dim, random_state=11)
        train_tfidf = tfidf.fit_transform(train['text'])
        X_train = svd.fit_transform(train_tfidf)
        y_train = train['genre'].values

        # transform test data using tfidf and svd objects fitted with train data
        test_tfidf = tfidf.transform(test['text'])
        X_test = svd.transform(test_tfidf)
        y_test = test['genre'].values

        # save train/test data and objects
        self.tfidf = tfidf
        self.svd = svd
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test


    def model_selection(self):
        parameters = {'n_estimators': [50, 100],
                      'class_weight': ['balanced', None],
                      'max_depth': [20, 35, 50]
                     }
        gridsearch = GridSearchCV(RandomForestClassifier(), parameters, cv=4, n_jobs=-1, verbose=10)
        gridsearch.fit(self.X_train, self.y_train)

        self.best_hyp_param = gridsearch.best_params_


    def train_classifier(self, hyp_param):
        if hyp_param is None:
            hyp_param = self.best_hyp_param

        movieclf = RandomForestClassifier(**hyp_param, random_state=15)
        movieclf.fit(self.X_train, self.y_train)
        self.movieclf = movieclf

        print('Test Score: ', self.movieclf.score(self.X_test, self.y_test))


    def save_model(self, mpath):
        dump(self.movieclf, mpath + 'movieclassifier.joblib')
        dump(self.tfidf, mpath + 'tfidf.joblib')
        dump(self.svd, mpath + 'svd.joblib')

    # util function to extract 1 top genre from movie genre list, if present
    def __get_genre_in_top(self, g_list, main_genres):
        top_genres = g_list.intersection(main_genres)

        if len(top_genres) > 0:
            return top_genres.pop()
        else:
            return 'no main genre'

    # util function to parse json genre list and map to plain text list
    def __get_genres_list(self, g_list):
        outlist = []
        for g in g_list:
            outlist.append(g['name'])

        return set(outlist)

def run(model_selection):
    movietrainer = MovieClassifierTrainer()

    movietrainer.extract_data('the-movies-dataset/movies_metadata.csv')
    movietrainer.prepare_features()

    if model_selection:
        movietrainer.model_selection()
        movietrainer.train_classifier()
    else:
        # precomputed hyper-parameters from IPYNB notebook "Train Classifier"
        precomputed_hyp_param = {'class_weight': None, 'max_depth': 20, 'n_estimators': 100}
        movietrainer.train_classifier(precomputed_hyp_param)

    movietrainer.save_model('models/')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Movie Classifier Training')
    parser.add_argument('--model-selection', dest='model_sel', action='store_true')
    parser.set_defaults(feature=False)

    args = parser.parse_args()
    run(args.model_sel)
