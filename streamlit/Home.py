import streamlit as st

st.title("My Cumulative Data Science App")
st.write("Luke Wilsen")

st.divider()

st.markdown(
    """
### Overview
This app demonstrates various data science and machine learning techniques applied to multiple datasets. Below is a brief summary of each dataset, and some practical applications.

### Data
- **Titanic**: Passenger data from the Titanic disaster, used for binary classification (survived or not).
- **Ames Housing**: Housing data from Ames, Iowa, between 2006 and 2010. I used different types of regression to predict house prices.
- **IMDB Movie Reviews**: Sentiment analysis of movie reviews, used for binary text classification (positive or negative).
- **MNIST Digits**: Handwritten digit images (0-9), used for image classification.
            
### Prediction
- **Titanic**: Passenger survival status is predicted based on Age, Sex, Fare, Number of Parents, Number of Children, Marriage Status, Number of Siblings, and Class (1st, 2nd, or 3rd).
- **Ames Housing**: House price of a home in Ames, Iowa, between 2006 and 2010 is predicted, using Quality, Condition, Exterior Quality, Basement SF, Central Air, Living Area, and Number of Fireplaces.
- **IMDB Movie Reviews**: The sentiment of a typed sentence is classified as positive or negative.
- **MNIST Digits**: You can draw a digit 0-9 and the model will attempt to recognize the digit.
"""
)
