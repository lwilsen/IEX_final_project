from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from nltk.stem import PorterStemmer

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, add_constant=0.0):
        self.add_constant = add_constant

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.log(X + self.add_constant)
    

porter = PorterStemmer()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]