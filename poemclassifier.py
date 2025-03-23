import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the datasets
train_data = pd.read_csv('/content/Poem_classification - train_data.csv')
test_data = pd.read_csv('/content/Poem_classification - test_data.csv')

# Data Cleaning
train_data['Genre'] = train_data['Genre'].str.strip()
train_data['Poem'] = train_data['Poem'].str.strip()
train_data = train_data.dropna()

test_data['Genre'] = test_data['Genre'].str.strip()
test_data['Poem'] = test_data['Poem'].str.strip()
test_data = test_data.dropna()

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to extract BERT embeddings
def get_bert_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

# Generate embeddings
X_train = get_bert_embeddings(train_data['Poem'].tolist())
y_train = train_data['Genre']

X_test = get_bert_embeddings(test_data['Poem'].tolist())
y_test = test_data['Genre']

# Split training data into train and validation sets
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_split, y_train_split)

# Evaluate on validation set
y_val_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Classification Report (Validation):\n", classification_report(y_val, y_val_pred))

# Test the model
y_test_pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("Classification Report (Test):\n", classification_report(y_test, y_test_pred))

# Predict genre for a new poem input
def predict_genre(poem):
    poem_embedding = get_bert_embeddings([poem])
    genre_prediction = model.predict(poem_embedding)
    return genre_prediction[0]

# Example
user_poem = input("Enter your poem: ")
predicted_genre = predict_genre(user_poem)
print(f"Predicted Genre: {predicted_genre}")