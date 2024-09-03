import re

import requests
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

import streamlit as st

nltk.download("punkt_tab")
nltk.download("stopwords")
st.title("Data Exploration")

if st.sidebar.button("Reset Page"):
    st.markdown(
        """
        <script>
        document.documentElement.scrollTop = 0;
        </script>
        """,
        unsafe_allow_html=True,
    )


def housing(df):
    st.write("---")
    st.write(
        """
    # Housing Price Linear Regression

    This data came from the assessor's office of Ames Iowa, and contains information used to computing 
            assessed values for individual residential properties sold in ames, IA from 2006 to 2010.
        """
    )
    st.write("""---""")
    st.image("/app/pictures/Ames_Iowa.jpg", caption="Ames Iowa", use_column_width=True)
    st.write(
        """        
    I chose to look at a subsection of the entire dataset, selecting only the following columns:
    """
    )

    st.write(
        """
                
        | Column Name    | Description                                                                     |
        |----------------|---------------------------------------------------------------------------------|
        | Overall Qual   | Rates the overall material and finish of the house. (ordinal 1-10)              |
        | Overall Cond   | Rates the overall condition of the house. (ordinal 1-10)                        |
        | Gr Liv Area    | Above grade (ground) living area square feet. (continuous)                      |
        | Central Air    | Central air conditioning. (Nominal)                                             |
        | Total Bsmt SF  | Total square feet of basement area. (continuous)                                |
        | SalePrice      | Sale price of the house. (continuous)                                           |
        | Fireplaces     | The number of fireplaces. (discrete)                                            |
        | Exter Qual     | Evaluates the quality of the material on the exterior.                          |
        ---
        """
    )

    st.subheader("""Queried Data""")
    st.dataframe(df)
    st.divider()
    st.write(
        "When I was exploring the data, I found that our outcome variable, *saleprice* was very skewed. This isn't good because a linear model requires non-skewed data to work properly!"
    )
    st.image(
        "/app/pictures/Housing_Normalization.png",
        caption="Normalization",
        use_column_width=True,
    )
    st.write(
        """
- You can see below that when the data isn't normalized the errors aren't **random**
    - This means your predictions are going to be wrong in a predictable way, and we can find and eliminate that pattern of error"""
    )
    st.image("/app/pictures/Non_Normal_Residuals.png", use_column_width=True)
    st.write(
        """
- However, after normalization (below), the errors are more random (I promise this is a good thing)
    - It means that we've extracted all of the patterns (predictability) out of the data we can!"""
    )
    st.image("/app/pictures/Normalized_residuals.png", use_column_width=True)
    st.divider()
    st.write(
        """I also tried to change different parameters of my best model, to see if I could fine tune it further, with limited success"""
    )
    st.image("/app/pictures/LR_val_curve.png", use_column_width=True)
    st.write(
        """
- You can see a slight slope in both lines (indicating better performance), but not by a practically significant amount"""
    )
    st.divider()
    st.subheader("Conclusions")
    st.image("/app/pictures/LR_coefficient_estimates.png", use_column_width=True)
    st.write(
        """
**Important predictors**
  
- Central Air
- Exter Qual
- Overall Qual
    - To see how a single parameter impacts price, **multiply the parameter by $100,00**
        - Ex. According to the model, **having central air** corresponds to an **increase in price by 0.243 hundred thousand dollars ($24,300)**.
    """
    )


def titanic(df1):

    st.subheader("Queried Data")
    st.dataframe(df1)
    st.write(
        """
    ---       
    ## Columns and what they represent
            
    | Column Name    | Description                                                                     |
    |----------------|---------------------------------------------------------------------------------|
    | PassengerId    | A unique numerical identifier assigned to each passenger.                         |
    | Survived       | Survival status of the passenger (0 = No, 1 = Yes).                              |
    | Pclass         | The passenger's ticket class (1 = 1st Class, 2 = 2nd Class, 3 = 3rd Class).   |
    | Sex            | The passenger's gender (male, female).                                         |
    | Age            | The passenger's age in years. Fractional values may exist for younger children. |
    | SibSp          | The number of siblings or spouses traveling with the passenger.                   |
    | Parch          | The number of parents or children traveling with the passenger.                   |
    | Fare           | The price the passenger paid for their ticket.                                  || class_1 - class_3          | The Passenger's room class.                                    |   
    """
    )
    st.divider()
    st.subheader("Data Exploration")
    st.write(
        "I wanted to visually explore the relationship between the different variables, so I used a heatmap"
    )
    st.image("/app/pictures/Titanic_heatmap.png", use_column_width=True)
    st.write(
        "- From the heatmap *survived* row, you can see that being **male** had a negative impact on survival, and that the **fare** you paid and the **class** you were in had either a negative or positive impact"
    )
    st.write(
        """I also divided the passengers into male and female to look at the differences in survival"""
    )
    st.image("/app/pictures/Titanic_pass_breakdown_sex.png", use_column_width=True)
    st.write(
        """
- You can see from the above graphs, that though **most of the passengers were men, most of the survivors were female**
    - Therefore it's likely that being male negatively impacted your chances of survival
             
I also wanted to look at the fare distrubution seperated by survival status
"""
    )
    st.image("/app/pictures/Titanic_violin.png", use_column_width=True)
    st.write(
        """
- The mean (dotted) lines in the above plot show us that the average *fare* paid was higher for those who survived,
             than for those that died
    - Therefore, it's also likely that the **more you paid, the better your chances of survival** were
        - Class could also have paid a role too, because if you paid more, you were more likely to be in first class, 
             which was higher up in the ship, also making survival more likely
"""
    )
    st.write(
        "I also tried a non-linear clustering method to see if there were two distinct clusters in the data"
    )
    st.image("/app/pictures/Titanic_DBSCAN.png", use_column_width=True)
    st.write(
        """
- Visually, you can see that there are two majority groups and one very small minority group
    - My hypothesis about this is that the two majority groups are:
             
    **1.** Did not survive, poor, 3rd or 2nd class and possibly male
                
    **2.** Survived, rich, 1st class, and most likely female
                
    **3.** Survived, rich, 1st class and most likely male
    - This is just based on what I was seeing as the biggest clusters, and the evidence from the violin plot above
"""
    )
    st.divider()
    st.subheader("Conclusions")
    st.write(
        """
Overall the insights that I gained from this analysis were:
- Being male, 
- Being poor 
- And being in 3rd or 2nd class 
             
**Negatively** impacted odds of survival. These conclusions back up most of what we already knew about the sinking of the Titanic.
"""
    )


def preprocessor(text):
    text = re.sub("<[^>]*>", "", text)
    emoticons = re.findall(r"(?::|;|=)(?:-)?(?:\)|\(|D|P)", text)
    text = re.sub(r"[\W]+", " ", text.lower()) + " ".join(emoticons).replace("-", "")

    return text


def movies(df2):
    st.write("**IMDB Dataframe**")
    st.dataframe(df2)
    st.divider()
    movie_1 = df2.iloc[1, 0]
    st.text_area("Unprocessed Review", movie_1, height=200)
    processed_review = preprocessor(movie_1)
    st.text_area("Cleaned Review", processed_review, height=200)
    tokenized_review = word_tokenize(processed_review)
    st.text_area("Tokenized Review", tokenized_review, height=250)
    stop = stopwords.words("english")
    st.text_area(
        "Processed Review",
        [word for word in tokenized_review if word not in stop],
        height=200,
    )
    st.divider()
    st.write("**Models**")
    st.write(
        """
I used two models:
1. Logistic Regression Classifier
    - This was the better performer
2. Out of Core Learning with a Stochastic Gradient Descent Classifier
    - This model iterates through "batches" of the total dataset, and updates itself during the process
        - This method is better for larger datasets and more complicated patterns
"""
    )
    st.divider()
    st.write("**Confusion Matrix**")
    st.write(
        """
- Confusion Matrices are often used for visually evaluating sentiment analysis
- You can see that my best model was better at predicting positive sentiments than negative sentiments
    - There are more incorrect "True Negative" reviews, than incorrect "True Positive" reviews
"""
    )
    st.image(
        "/app/pictures/NLP_Confusion.png",
        caption="Confusion Matrix of model predictions",
        use_column_width=True,
    )
    st.divider()
    st.subheader("Take-away")
    st.write(
        """
- There aren't really any *insights* to take away from this dataset, **but**, we do now have a cool model that can analyze the sentiment of any text passed to it!
"""
    )


def mnist():
    st.write("**Examples of MNIST Handwritten Digits**")
    st.image(
        "/app/pictures/MNIST_digits.png", caption="MNIST DIGITS", use_column_width=True
    )
    st.header("Convolutional Neural Network Using *Tensorflow*")
    st.markdown(
        """
I used a Convolutional Neural Network (CNN) to classify the images contained in the MNIST dataset, using both Pytorch and Tensorflow. Here is a brief explanation of each layer in the CNN model and its role in the classification process:
"""
    )

    st.subheader("Neural Network Architecture Explained")

    st.write(
        """
### 1. Flatten Layer
- **What It Does**: This layer takes the image, which is originally 28 by 28 pixels, and stretches it out into a single line with 784 values. This step is needed so that the image data can be processed by the next layers.
"""
    )

    st.write(
        """
### 2. Decision-Making Layer 1 (Dense Layer 1)
- **What It Does**: This layer has 128 decision-making units (neurons). These units learn to detect important features from the stretched-out image data. They use a method called "ReLU" that helps the network identify complex patterns and relationships within the data.
"""
    )

    st.write(
        """
### 3. Final Decision (Output Layer)
- **What It Does**: The final layer has 10 units, one for each possible digit (0-9). This layer makes the final decision by processing the data into 10 possible outcomes (one for each digit). Although this layer uses a "linear" method at first, a step called "softmax" is applied later to convert the results into probabilities, which tells us how confident the model is in each choice.
"""
    )
    st.subheader("Model Compilation")
    st.write(
        """
### 1. Optimizer: Adam
- **What It Does**: The Adam optimizer helps the model learn from the training data by adjusting the model's internal settings (weights) based on the data it sees. It combines the benefits of two other methods, which makes learning faster and more efficient.
"""
    )

    st.write(
        """
### 2. Loss Function: Sparse Categorical Crossentropy
- **What It Does**: This function is like a guide that helps the model learn to make better predictions. It compares the model’s guesses with the correct answers (the target labels), tells the model how far off it is, and in which direction to go to get better.
"""
    )

    st.write(
        """
### 3. Evaluation Metric: Accuracy
- **What It Does**: Accuracy tells us how well the model is performing by calculating the percentage of correctly guessed images. This metric tells us how well the model did overall, and at each step through its learning process.
"""
    )

    st.write("---")
    st.write("Model Training and Validation")
    st.image(
        "/app/pictures/Screenshot 2024-05-29 at 7.53.46 PM.png",
        caption="Train and Validation Accuracy through Epoch Progression",
    )

    st.write("Graphs of model performance through epoch progression")
    st.image(
        "/app/pictures/Screenshot 2024-05-29 at 7.54.14 PM.png",
        caption="Model Performance",
    )


data_option = st.radio(
    "Select Data:", ("Housing", "Titanic", "Movie", "MNIST"), key="data_option"
)

query = ""
if data_option == "Titanic":
    query = "SELECT * FROM titanic LIMIT 30;"
elif data_option == "Housing":
    query = "SELECT * FROM housing LIMIT 30;"
elif data_option == "Movie":
    query = "SELECT * FROM movies LIMIT 30;"
elif data_option == "MNIST":
    query = "-- MNIST is non-queriable from the SQL database"

query = st.text_area(label="**Enter your SQL query here:**", value=query)

if st.button("Submit"):
    response = requests.post(
        "http://flask_route:5001/query", json={"query": query}, timeout=10
    )
    if response.status_code == 200:
        try:
            result = response.json()

            data = result.get("Data", [])
            columns = result.get("Columns", [])
            dataframe = pd.DataFrame(data, columns=columns)
            if data_option == "Housing":
                housing(dataframe)
            elif data_option == "Titanic":
                titanic(dataframe)
            elif data_option == "Movie":
                movies(dataframe)

        except requests.exceptions.JSONDecodeError:
            st.error("Error: The response is not in JSON format.")
            st.write("Response content:", response.text)
    elif data_option == "MNIST":
        mnist()

    else:
        st.error(f"Error: Received status code {response.status_code}")
        st.write("Response content:", response.text)
