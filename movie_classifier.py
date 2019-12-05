from joblib import load
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import argparse

stop_words = nltk.corpus.stopwords.words('english')
stemmer = nltk.stem.snowball.EnglishStemmer()

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (stemmer.stem(w) for w in analyzer(doc))


class MovieClassifier:

    def __init__(self, modelpath='models/'):
        self.mpath = modelpath

    def load_models(self):
        self.movieclf = load(self.mpath + 'movieclassifier.joblib')
        self.tfidf = load(self.mpath + 'tfidf.joblib')
        self.svd = load(self.mpath + 'svd.joblib')

    def get_genre(self, title, description):
        input_text = [title + ' ' + description]

        x = self.tfidf.transform(input_text)
        X = self.svd.transform(x)

        genre = self.movieclf.predict(X)[0]

        return {'title': title,
                'overview': description,
                'genre': genre}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Movie Classifier')
    parser.add_argument('--title', type=str, help='Movie title')
    parser.add_argument('--description', type=str, help='Movie description')

    args = parser.parse_args()

    if not args.title or not args.description:
        print('Title and description are mandatory !')

    else:
        movies_clf = MovieClassifier()
        movies_clf.load_models()
        result = movies_clf.get_genre(args.title, args.description)

        print(result)
