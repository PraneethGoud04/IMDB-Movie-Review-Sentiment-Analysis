import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
import re

from tensorflow.keras.preprocessing.sequence import pad_sequences


model = tf.keras.models.load_model("Imdb-movie-Sentiment-analysis.keras")

with open("tokenizer.pkl","rb") as f:
    tokenizer = pickle.load(f)

with open("stopwords.pkl","rb") as f:
    stop_words = pickle.load(f)


MAX_LEN = 250


def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    tokens = text.split()

    tokens = [word for word in tokens if word not in stop_words]

    return tokens


def predict_sentiment(review):

    tokens = clean_text(review)

    sequence = tokenizer.texts_to_sequences([tokens])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="pre",
        truncating="post"
    )

    probability = float(model.predict(padded, verbose=0)[0][0])

    if probability >= 0.5:
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    return sentiment, probability


st.set_page_config(page_title="IMDB Sentiment Analysis",page_icon="🎬",layout="centered")

st.title("DeepSenti: AI-Based Movie Review Sentiment Analysis")



review = st.text_area(
    "Enter Review",
    height=100
)

if st.button("Predict Sentiment"):

    if review.strip()=="":

        st.warning("Please enter a review.")

    else:

        sentiment, probability = predict_sentiment(review)
        
        
        confidence = probability if sentiment == "Positive" else 1 - probability
        
        if sentiment == "Positive":
            st.success("Positive Review")
        else:
            st.error("Negative Review")
            
            
    st.metric("Confidence",f"{confidence*100:.2f}%")
            
    st.progress(confidence)