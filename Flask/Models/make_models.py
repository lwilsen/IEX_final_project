from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.stem.porter import PorterStemmer
from joblib import Memory

memory = Memory(location='./cache')

#Housing regression model creation
from final_project.Flask.utils import LogTransformer
    
ridge_pipeline = Pipeline([
    ('log_transformer', LogTransformer(add_constant=1)),
    ('ridge_regression', Ridge(random_state=42))
], memory = memory)

with open('ridge_pipe.pkl', 'wb') as f:
    pickle.dump(ridge_pipeline, f)


# NLP Models n stuff
porter = PorterStemmer()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(solver='liblinear', random_state=42)),], memory=memory)

best_params = {'clf__C': 10.0, 'clf__penalty': 'l2', 'vect__ngram_range': (1, 1), 'vect__stop_words': None, 'vect__tokenizer': tokenizer_porter}
lr_tfidf.set_params(**best_params)

