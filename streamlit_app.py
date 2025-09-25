import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# Load the saved Keras model
model = tf.keras.models.load_model('sentiment_model.h5')

# Load the saved LabelEncoder and Tokenizer objects
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Define the maximum sequence length used during training
MAX_LEN = 60

# Create a Streamlit title for the application
st.title('Sentiment Analysis of Financial Headlines')

# Add a text area for user input
user_input = st.text_area("Enter a financial headline:")

# Create a button to trigger the sentiment prediction
if st.button('Analyze Sentiment'):
    if user_input:
        # Tokenize and pad the user input
        seq = tokenizer.texts_to_sequences([user_input])
        pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

        # Make a prediction
        probs = model.predict(pad)
        pred_indices = np.argmax(probs, axis=1)

        # Inverse transform the prediction to get the sentiment label
        pred_label = le.inverse_transform(pred_indices)

        # Display the predicted sentiment
        st.write(f"Predicted Sentiment: {pred_label[0]}")

        # (Optional) Display the prediction probabilities
        st.write(f"Probabilities: {probs[0]}")
    else:
        st.write("Please enter a headline to analyze.")
