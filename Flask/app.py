import sqlite3
import pickle

from flask import Flask, request, jsonify

from flask_cors import CORS
import tensorflow as tf
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("wordnet")

app = Flask(__name__)
CORS(app)  # Need to research more about CORS (and how to use it properly)


@app.route("/")
def home():
    """Checks to make sure the app is running properly."""
    return "Hello World! This is the home page of the app!"


# Build Query tool (not including mnist for now)


def query_database(qery):
    """Queries the SQLite database that hosts all of the data used in the project.
    
    The three datasets are the Titanic dataset, the Ames Iowa housing dataset,
    and the IMDB movie review dataset. The MNIST dataset couldn't be converted to a 
    SQL database without more trouble than it was worth."""

    try:
        conn = sqlite3.connect("/app/final_project.db")
        cursor = conn.cursor()
        cursor.execute(qery)
        columns = [description[0] for description in cursor.description]
        data = cursor.fetchall()

        conn.close()
        return {"Columns": columns, "Data": data}
    except sqlite3.Error as e:
        return {"error": str(e)}


# Prediction Functions/models

## Import Models

### Best for Titanic

with open("Models/svc_pipeline.pkl", "rb") as mod:
    svc_pipe = pickle.load(mod)

### Best for LR on Housing
with open("data.pkl", "rb") as f:
    housing_df = pickle.load(f)

housing_df["Log_saleprice"] = np.log(housing_df["SalePrice"])
housing_df = housing_df.iloc[:, list(range(0, 7)) + [8]]

LOG_TARGET = "Log_saleprice"
log_feats = housing_df.columns[(housing_df.columns != LOG_TARGET)]
# Now housing only has feature columns and log target column

X_log = housing_df[log_feats]

y_log = housing_df[LOG_TARGET]

Xl_train, Xl_test, yl_train, yl_test = train_test_split(
    X_log, y_log, test_size=0.3, random_state=123
)

ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(Xl_train, yl_train)

porter = PorterStemmer()


def tokenizer_porter(text):
    """Creates a new tokenizer using the porter stemmer."""

    return [porter.stem(word) for word in text.split()]


# NLP model is the lr_tfidf


with open("Models/nlp_model.pkl", "rb") as mod2:
    nlp_mod = pickle.load(mod2)


def get_wordnet_pos(treebank_tag):
    """Tags each word with it's word type (noun, adjective, verb)."""

    if treebank_tag.startswith("J"):
        result = wordnet.ADJ
    if treebank_tag.startswith("V"):
        result = wordnet.VERB
    if treebank_tag.startswith("N"):
        result = wordnet.NOUN
    if treebank_tag.startswith("R"):
        result = wordnet.ADV
    else:
        result = wordnet.NOUN
    return result


def preprocess_text(text):
    """Performs the preprocessing needed for sentiment analysis."""

    tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    processed_tokens = []
    for word, tag in pos_tags:
        lemmatized_word = lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        stemmed_word = stemmer.stem(lemmatized_word)
        processed_tokens.append(stemmed_word)
    return " ".join(processed_tokens)


nlp_mod.set_params(vect__tokenizer=tokenizer_porter)

### MNIST model

mnist_model = tf.keras.models.load_model("Models/mnist_model.keras")

# Functions that recieve POST requests


@app.route("/query", methods=["POST"])
def query():
    """Queries the database and provides the data back to the streamlit app."""

    try:
        query_response = request.json.get("query")
        if not query_response:
            return jsonify({"error": "No Query Provided"}), 400
        data = query_database(query_response)
        return jsonify(data)
    except sqlite3.Error as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict_titanic", methods=["POST"])
def predict_titanic():
    """Predicts survival status given user input data."""

    data = request.json
    df = pd.DataFrame([data])
    prediction = svc_pipe.predict(df)
    return jsonify({"Survived": int(prediction)})


@app.route("/predict_housing", methods=["POST"])
def predict_housing():
    """Predicts house price given user input data."""


    data = request.json
    df = pd.DataFrame([data])
    prediction = ridge.predict(df)
    return jsonify({"Sale_Price": float(prediction)})


@app.route("/predict_sentiment", methods=["POST"])
def predict_sentiment():
    """Predicts the sentiment given user input text."""


    data = request.json
    text = data.get("text", "")
    processed_text = preprocess_text(text)
    array = np.array([processed_text])
    pred = nlp_mod.predict(array)
    sentiment = "Positive" if pred[0] == 1 else "Negative"
    return jsonify({"sentiment": sentiment})


@app.route("/predict_mnist", methods=["POST"])
def predict_mnist():
    """Predicts digit given user input data."""
    
    data = request.json.get("image_data")
    processed_img = np.array(data).reshape((1, 28, 28, 1)) / 255.0
    prediction = mnist_model.predict(processed_img)
    predicted_digit = np.argmax(prediction)
    return jsonify({"digit": predicted_digit.tolist(), "weights": prediction.tolist()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
