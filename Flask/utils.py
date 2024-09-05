"""Used for the creation of a tokenizer for nlp processing."""

from nltk.stem import PorterStemmer

porter = PorterStemmer()


def tokenizer_porter(text):
    """Creates a new tokenizer using the porter stemmer."""

    return [porter.stem(word) for word in text.split()]
