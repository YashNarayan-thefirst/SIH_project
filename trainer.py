import tensorflow as tf
import os
import string
import nltk
import re
import concurrent.futures
import gpt_2_simple as gpt2
from collections import defaultdict
import json

# Set seed for reproducibility
tf.random.set_seed(42)

# Parameters
max_seq_length = 900
num_labels = 10**5
batch_size = 8
epochs = 10
target_data_size_mb = 0.01

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

def process_json_lines(lines):
    qa_pairs = []

    for line in lines:
        json_data = json.loads(line)  # Parse the line as a JSON object
        conversations = json_data.get("conversations", [])

        # Extract human questions and GPT-2 responses
        for i in range(1, len(conversations), 2):
            if conversations[i - 1]["from"] == "human" and conversations[i]["from"] == "gpt":
                question = conversations[i - 1]["value"]
                answer = conversations[i]["value"]
                qa_pairs.append((question, answer))

    return qa_pairs

def load_or_train_gpt2():
    # Check if the fine-tuned model exists
    if not os.path.exists("fine_tuned_gpt2"):
        sess = gpt2.start_tf_sess()
        gpt2.load_gpt2(sess, model_name='124M')

        # Fine-tune the GPT-2 model with your dataset
        fine_tuned_model = gpt2.finetune(sess,
                                        dataset='Puffin.jsonl',
                                        model_name='124M',
                                        steps=100,
                                        reuse = True
                                        )
        print("GPT-2 training completed and fine-tuned model saved.")
    else:
        print("Fine-tuned model found. Loading fine-tuned GPT-2 model...")
        sess = gpt2.start_tf_sess()
        gpt2.load_gpt2(sess, model_name='fine_tuned_gpt2')

    return sess

if __name__ == '__main__':
    input_data = []

    # Process the text lines in the JSON file with a data size limit of 10 MB
    target_data_size_bytes = target_data_size_mb * 1024 * 1024  # Convert to bytes
    pool = concurrent.futures.ProcessPoolExecutor()  # Use ProcessPoolExecutor for parallel processing
    futures = []

    with open('Puffin.jsonl', 'r', encoding='utf-8') as jsonl_file:
        current_data_size_bytes = 0
        lines = []

        for line in jsonl_file:
            if current_data_size_bytes >= target_data_size_bytes:
                break

            lines.append(line)
            current_data_size_bytes += len(line.encode('utf-8'))

            if len(lines) >= 100:  # Process lines in batches of 100 (adjust as needed)
                future = pool.submit(process_json_lines, lines)
                futures.append(future)
                lines = []

        if len(lines) > 0:
            future = pool.submit(process_json_lines, lines)
            futures.append(future)

    for future in concurrent.futures.as_completed(futures):
        input_data.extend(future.result())

    print("Number of input samples:", len(input_data))

    # Load or train the GPT-2 model
    sess = load_or_train_gpt2()
