import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
data = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

# Encode labels
encoder = LabelEncoder()
data['label'] = encoder.fit_transform(data['label'])

X = data['message']
y = data['label']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenizer
vocab_size = 5000
max_len = 100

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

train_seq = tokenizer.texts_to_sequences(X_train)
X_train_pad = pad_sequences(train_seq, maxlen=max_len, padding='post')

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 64, input_length=max_len),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_pad, y_train, epochs=3, batch_size=32, verbose=0)

# Streamlit UI
st.title("📧 Spam Email Detector")
st.write("Enter email text below to check if it is Spam or Not Spam")

user_input = st.text_area("Enter Email Text")

if st.button("Predict"):
    seq = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    prediction = model.predict(padded)[0][0]

    if prediction > 0.5:
        st.error("🚨 This is Spam Email")
    else:
        st.success("✅ This is Not Spam")
