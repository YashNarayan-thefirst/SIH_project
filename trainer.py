import json
import numpy as np
import tensorflow as tf
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
import joblib

# Define the maximum data size limit in bytes (150 MB)
max_data_size_limit = 150 * 1024 * 1024  # 150 MB in bytes

# Initialize label encoder for handling class labels
label_encoder = LabelEncoder()

# Initialize a Keras tokenizer
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")

# Initialize lists for storing input and output data
input_data = []
output_data = []
total_data_size =0
with open('dataset.jsonl', 'r', encoding='utf-8') as jsonl_file:
    for line in jsonl_file:
        item = json.loads(line)
        input_text = item["input"]
        output_text = item["output"]

        # Calculate the size of the current chunk in bytes
        chunk_size = len(input_text.encode('utf-8')) + len(output_text.encode('utf-8'))

        # Check if adding the current chunk exceeds the data size limit
        if total_data_size + chunk_size > max_data_size_limit:
            print("Data size limit exceeded. Training on current chunks...")
            # Train the model on the current chunks
            if input_data:
                # Encode the class labels
                y_train = label_encoder.fit_transform(output_data)
                
                # Tokenize and preprocess input data
                tokenizer.fit_on_texts(input_data)
                X_train = tokenizer.texts_to_sequences(input_data)
                X_train = pad_sequences(X_train, maxlen=100, padding='post')
                
                # Define a simple LSTM model
                model = Sequential()
                model.add(Embedding(5000, 64, input_length=100))
                model.add(LSTM(100))
                model.add(Dense(len(set(y_train)), activation='softmax'))

                model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

                model.fit(X_train, y_train, epochs=10, batch_size=64)
                
                input_data.clear()
                output_data.clear()
                print("Finished training on the current chunks.")

        # Add the current chunk to the lists
        input_data.append(input_text)
        output_data.append(output_text)
        total_data_size += chunk_size

# Train the model on any remaining chunks
if input_data:
    print("Training on remaining chunks...")
    # Encode the class labels
    y_train = label_encoder.fit_transform(output_data)

    # Tokenize and preprocess input data
    tokenizer.fit_on_texts(input_data)
    X_train = tokenizer.texts_to_sequences(input_data)
    X_train = pad_sequences(X_train, maxlen=100, padding='post')

    # Define a simple LSTM model
    model = Sequential()
    model.add(Embedding(5000, 64, input_length=100))
    model.add(LSTM(100))
    model.add(Dense(len(set(y_train)), activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=64)

    print("Finished training on the remaining chunks.")

# Save the trained model
model_filename = "text_classification_model.h5"
model.save(model_filename)

print(f"Model saved as {model_filename}")
