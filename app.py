import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained LSTM model
@st.cache_resource
def load_lstm_model():
    return load_model('next_word_lstm.h5')

# Load tokenizer
@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pickle', 'rb') as handle:
        return pickle.load(handle)

model = load_lstm_model()
tokenizer = load_tokenizer()

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            return word
    return None

# Streamlit UI
st.title("🔮 Next Word Prediction with LSTM")

input_text = st.text_input("Enter a sequence of words", "To be or not to")

if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    if next_word:
        st.success(f"**The next word is:** {next_word}")
    else:
        st.error("Couldn't predict the next word. Try another input!")

