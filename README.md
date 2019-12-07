# ML TrueLayer Challenge

## Installation Instructions

Create a new environment using Anaconda (download and install from [here](https://www.anaconda.com/distribution/)):

```
conda create -n truelayer python=3.7
```

Activate the environment:

```
conda activate truelayer
```

Install packages from requirements.txt:

```
conda install --file requirements.txt
```

Install NLTK stopwords.Run the Python interpreter and type the commands:

```python
import nltk
nltk.download()
```

and select stopwords form the interface.

## How to reproduce results

IPython Notebooks are used to explore data, features and possible solutions: they can be run to generate images and exploratory tables.

Python scripts contain the final version of the code:
1. ```python train.py```: load and preprocess MovieLens public dataset, train the model and store it in models/. Using  ```--model-selection``` it additionally performs hyper-parameters grid-search, otherwise a set of precomupted optimal hyper-parameters are used;
2. ```python movie_classifier.py --title title --description description```: load model and precomputed objects generated using previous script and output the predicted genre for given title and description.

## Description

The repository implements a machine learning pipeline to train and use a movie classifier.

The code is implemented in Python, because it is a very flexible programming language that allows to easily define ML scripts. It also has already implemented many ML and NLP libraries to ease the whole process definition: in particular, scikit-learn and nltk are used to solve this problem.

NLP techniques are applied to extract numerical features from raw text:
1. TF-IDF is run on training corpus: each word of the description and the title is mapped to a weight, thus generating a numerical feature vector for each movie. This technique, even if not the state of the art in NLP, provides a simple and fast mapping of text to numerical features, which can be easily exploited by any kind of classifiers;
2. SVD is applied to previous output to reduce dimensionality. Since TFIDF features space is defined with one dimension for each distinct word in the corpus, while each movie has only few descriptive words, movie vectors are extremely sparse: SVD decomposition allows to easily reduce number of dimensions in sparse scenarios. Number of new features is manually selected based on overall explained variance (explored in Notebook).
3. Final features vectors are used to train a Random Forest classifier: this ensemble method has been selected because it is extremely fast to train, and produce reasonably good results.
