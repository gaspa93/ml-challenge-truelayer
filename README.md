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
