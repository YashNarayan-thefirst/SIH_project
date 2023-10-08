import json
import nltk
import tensorflow as tf
import re
import numpy as np
import os
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from keras import mixed_precision
import string
from collections import defaultdict

mixed_precision.set_global_policy("mixed_float16")


max_data_size_limit = 0.05 * 1024 * 1024 

# Initialize variables to keep track of data size
total_data_size = 0
num_labels = 10**4  # Specify the number of labels

# Initialize NLTK stopwords
nltk.download('stopwords')
global stop_words
stop_words = set(nltk.corpus.stopwords.words('english'))
translator = str.maketrans('', '', string.punctuation)
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
    # Convert text to lowercase and remove special characters, punctuation, and extra whitespace
    text = text.lower()
    text = text.translate(translator)
    # Tokenize the text into words
    words = text.split()
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    # Join the words back into a single string
    preprocessed_text = ' '.join(words)
    return preprocessed_text

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

# Create a neural network model using RoBERTa
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = TFRobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels)

# Compile the model
optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# Function to process a single JSON line
def process_json_line(line, input_data, data_labels_list):
    json_data = json.loads(line)
    input_text = json_data["input"]
    output_text = json_data["output"]

    # Preprocess and tokenize the input text
    input_text = preprocess_text(input_text)
    input_tokenized = tokenizer(input_text, padding=True, truncation=True, return_tensors="tf", max_length=128)

    if "input_ids" in input_tokenized:
        input_data.append(input_tokenized["input_ids"][0].numpy())

    # Extract and preprocess labels from the output text
    current_labels = extract_labels_from_output(output_text)
    data_labels_list.extend([map_labels_to_numeric(label) for label in current_labels])

def process_json_file(file_path):
    input_data = []
    data_labels_list = []

    with open(file_path, 'r', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            process_json_line(line, input_data, data_labels_list)

    return input_data, data_labels_list

if __name__ == '__main__':
    input_data = multiprocessing.Manager().list()
    data_labels_list = multiprocessing.Manager().list()

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_json_file, ['dataset.jsonl']))

    for result in results:
        input_data.extend(result[0])
        data_labels_list.extend(result[1])

    print("Number of input samples:", len(input_data))
    print("Number of label samples:", len(data_labels_list))

    # Pad or truncate input sequences to a fixed length
    max_seq_length = 128
    x_train = tf.keras.preprocessing.sequence.pad_sequences(input_data, maxlen=max_seq_length, padding="post", truncating="post", dtype="int32")
    
    # Ensure that x_train and y_train have the same number of samples
    min_samples = min(len(x_train), len(data_labels_list))
    x_train = x_train[:min_samples]
    y_train = data_labels_list[:min_samples]
    y_train = np.array(y_train,dtype = np.int32)
    print("Shape of x_train:", x_train.shape)
    print("Length of y_train:", y_train.shape)

    model.fit(x_train, y_train, epochs=50, batch_size=64)  

    try:
        model.save_pretrained(os.path.join(os.getcwd(), "model"))
        tokenizer.save_pretrained(os.path.join(os.getcwd(), "model"))
        print("Model and tokenizer saved in the 'model' directory:", os.path.join(os.getcwd(), "model"))
    except Exception as e:
        print("An error occurred while saving the model:", str(e))
