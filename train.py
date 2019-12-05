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

stop_words = nltk.corpus.stopwords.words('english')
stemmer = nltk.stem.snowball.EnglishStemmer()

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (stemmer.stem(w) for w in analyzer(doc))


class MovieClassifierTrainer:

    def __get_main_genre(self, g_list):
        if len(g_list) > 0:

            if g_list[0]['name'] in main_genres:
                return g_list[0]['name']
            else:
                return 'no main genre'
        else:
            return 'no genre'


    def extract_data(self, source):
        movies = pd.read_csv(source)

        # transform from string to objects
        movies['genres'] = movies['genres'].apply(lambda x: np.nan if pd.isnull(x) else ast.literal_eval(x))

        # extract main genres list
        genres = []
        for m in movies['genres']:
            if m != []:
                genres.append(m[0])

        genre_list = pd.DataFrame(genres)['name'].value_counts()
        genre_list = genre_list.drop(['Odyssey Media', 'Aniplex', 'Carousel Productions'])
        main_genres = genre_list.keys().tolist()

        # define dataset for classifier
        target = movies[['id', 'title', 'overview', 'genres']]

        # remove movies without overview or title
        target = target[(~target['overview'].isnull()) & (~target['title'].isnull())]

        # extract only the first genre to ease the classifier training
        target['genre'] = target['genres'].apply(lambda x: __get_main_genre(x))
        target = target.drop('genres', axis=1)

        # remove movies that do not have genre or do not belong to main list
        target = target[~target['genre'].isin(['no genre', 'no main genre
        target.to_csv('data.csv', index=None)

        self.data = target

    def prepare_features(self, svd_dim=600):

        # preprocessing: define input text as concatenation of title and overview
        data['text'] = data.apply(lambda x: x['title'] + ' ' + x['overview'], axis=1)

        # split train and test data
        train, test = train_test_split(data, test_size=0.1, random_state=17)

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
        # model selection
        parameters = {'n_estimators': [50, 100],
                      'class_weight': ['balanced', None],
                      'max_depth': [20, 35, 50]
                     }
        gridsearch = GridSearchCV(RandomForestClassifier(), parameters, cv=4, n_jobs=-1, verbose=10)
        gridsearch.fit(self.X_train, self.y_train)

        self.best_hyp_param = gridsearch.best_params_


    def test_classifier(self):
        movieclf = RandomForestClassifier(**self.best_hyp_param, random_state=15)
        movieclf.fit(self.X_train, self.y_train)
        self.movieclf = movieclf

        print('Test Score: ', self.movieclf.score(self.X_test, self.y_test))


    def save_model(self, mpath):
        dump(self.movieclf, mpath + 'movieclassifier.joblib')
        dump(self.tfidf, mpath + 'tfidf.joblib')
        dump(self.svd, mpath + 'svd.joblib')

def run():
    movietrainer = MovieClassifierTrainer()
    movietrainer.extract_data('the-movies-dataset/movies_metadata.csv)
    movietrainer.prepare_features()
    movietrainer.model_selection()
    movietrainer.test_classifier()
    movietrainer.save_model()

if __name__ == '__main__':
    run()
