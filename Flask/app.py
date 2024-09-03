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
CORS(app)


@app.route("/")
def home():
    return "Hello World! This is the home page of the app!"


# Build Query tool (not including mnist for now)


def query_database(qery):
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
svc_pipe = pickle.load(open("Models/svc_pipeline.pkl", "rb"))

### Best for LR on Housing
with open("data.pkl", "rb") as f:
    housing_df = pickle.load(f)

housing_df["Log_saleprice"] = np.log(housing_df["SalePrice"])
housing_df = housing_df.iloc[:, list(range(0, 7)) + [8]]

log_target = "Log_saleprice"
log_feats = housing_df.columns[(housing_df.columns != log_target)]
# Now housing only has feature columns and log target column

X_log = housing_df[log_feats]

y_log = housing_df[log_target]

Xl_train, Xl_test, yl_train, yl_test = train_test_split(
    X_log, y_log, test_size=0.3, random_state=123
)

ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(Xl_train, yl_train)

porter = PorterStemmer()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


# NLP model is the lr_tfidf
# nlp_model = pickle.load(open('Models/nlp_model.pkl','rb'))
# nlp_vect = pickle.load(open('Models/nlp_vect.pkl','rb'))
nlp_mod = pickle.load(open("Models/nlp_model.pkl", "rb"))


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def preprocess_text(text):
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


@app.route("/query", methods=["GET", "POST"])
def query():
    if request.method == "GET":
        return "THIS WILL BE WHERE QUERIES ARE SENT"
    if request.method == "POST":
        try:
            query_response = request.json.get("query")
            if not query_response:
                return jsonify({"error": "No Query Provided"}), 400
            data = query_database(query_response)
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)}), 500


@app.route("/predict_titanic", methods=["POST"])
def predict_titanic():
    data = request.json
    df = pd.DataFrame([data])
    prediction = svc_pipe.predict(df)
    return jsonify({"Survived": int(prediction)})


@app.route("/predict_housing", methods=["POST"])
def predict_housing():
    data = request.json
    df = pd.DataFrame([data])
    prediction = ridge.predict(df)
    return jsonify({"Sale_Price": float(prediction)})


@app.route("/predict_sentiment", methods=["POST"])
def predict_sentiment():
    data = request.json
    text = data.get("text", "")
    processed_text = preprocess_text(text)
    array = np.array([processed_text])
    pred = nlp_mod.predict(array)
    sentiment = "Positive" if pred[0] == 1 else "Negative"
    return jsonify({"sentiment": sentiment})


@app.route("/predict_mnist", methods=["POST"])
def predict_mnist():
    data = request.json.get("image_data")
    processed_img = np.array(data).reshape(1, 28, 28, 1) / 255.0
    prediction = mnist_model.predict(processed_img)
    predicted_digit = np.argmax(prediction)
    return jsonify({"digit": predicted_digit.tolist(), "weights": prediction.tolist()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
