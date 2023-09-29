import json
import nltk
import tensorflow as tf
import re
import numpy as np
import os
from transformers import ElectraTokenizer, TFElectraForSequenceClassification

# Define the maximum data size limit in bytes (10 MB)
max_data_size_limit = 100 * 1024 * 1024  # 10 MB in bytes

# Initialize variables to keep track of data size
total_data_size = 0
data_labels_list = []
num_labels = 10**5  # Specify the number of labels

# Initialize NLTK stopwords
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

# Define the label mapping dictionary as a global variable
label_mapping = {}
current_label_id = 0

def map_labels_to_numeric(label):
    global current_label_id, label_mapping
    if label not in label_mapping:
        label_mapping[label] = current_label_id
        current_label_id += 1
    return label_mapping[label]

def preprocess_text(text):
    return text

def extract_labels_from_output(output_text):
    labels = set()  # Use a set to ensure unique labels
    
    # Split the text into sentences based on line breaks
    sentences = output_text.split('\n')
    
    # Iterate through the sentences and extract labels
    for sentence in sentences:
        # Use a regular expression to match lines starting with a bullet point
        match = re.match(r'•\s*(.*)', sentence)
        if match:
            label = match.group(1).strip()  # Extract the label text
            labels.add(label)  # Add the label to the set
    
    return list(labels)

# Create a neural network model
tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-discriminator")
model = TFElectraForSequenceClassification.from_pretrained("google/electra-small-discriminator", num_labels=num_labels)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# Read and process the dataset
input_data = []
output_data = []

with open('dataset.jsonl', 'r', encoding='utf-8') as jsonl_file:
    for line in jsonl_file:
        json_data = json.loads(line)
        input_text = json_data["input"]
        output_text = json_data["output"]

        # Calculate the size of the current chunk in bytes
        chunk_size = len(input_text.encode('utf-8')) + len(output_text.encode('utf-8'))

        # Check if adding the current chunk exceeds the data size limit
        if total_data_size + chunk_size > max_data_size_limit:
            print("Data size limit exceeded. Preprocessing and training on current chunks...")
            # Extract labels from the current chunk and append to data_labels_list
            current_labels = extract_labels_from_output("\n".join(output_data))
            data_labels_list.extend([map_labels_to_numeric(label) for label in current_labels])
            data_labels = np.array(data_labels_list)
                
            # Ensure data_labels is defined, even if there are no labels
            if len(data_labels) == 0:
                data_labels = np.zeros(len(input_data))

            # Tokenize and pad the input data for training
            inputs = tokenizer(input_data, padding=True, truncation=True, return_tensors="tf", max_length=128)

            # Split the data into training and testing sets
            split_ratio = 0.8
            split_index = int(len(input_data) * split_ratio)
            inputs_train, inputs_test = inputs[:split_index], inputs[split_index:]
            data_labels_train, data_labels_test = data_labels[:split_index], data_labels[split_index:]

            # Train the neural network model on the current data
            model.fit(inputs_train, data_labels_train, epochs=50, batch_size=32, validation_data=(inputs_test, data_labels_test))

            # Clear the processed data arrays
            input_data = []
            output_data = []
            total_data_size = 0

            print("Cleared processed data arrays.")

        # Add the current data to the arrays
        input_data.append(input_text)
        output_data.append(output_text)
        total_data_size += chunk_size

# Train the model on any remaining data if the limit is not exceeded
if input_data:
    print("Training on remaining data...")

    # Extract labels from the remaining data and append to data_labels_list
    remaining_labels = extract_labels_from_output("\n".join(output_data))
    data_labels_list.extend([map_labels_to_numeric(label) for label in remaining_labels])
    data_labels = np.array(data_labels_list)

    # Ensure data_labels is defined, even if there are no labels
    if len(data_labels) == 0:
        data_labels = np.zeros(len(input_data))

    # Tokenize and pad the input data for remaining data
    inputs_train_remaining = tokenizer(input_data, padding=True, truncation=True, return_tensors="tf", max_length=128)

    # Train the neural network model on the remaining data
    model.fit(inputs_train_remaining, data_labels, epochs=50, batch_size=32)

    try:
        model.save_pretrained(os.path.join(os.getcwd(), "model"))
        tokenizer.save_pretrained(os.path.join(os.getcwd(), "model"))
        print("Model and tokenizer saved in the 'model' directory:", os.path.join(os.getcwd(), "model"))
    except Exception as e:
        print("An error occurred while saving the model:", str(e))
