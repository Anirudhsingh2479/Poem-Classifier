from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the datasets
train_data = pd.read_csv('./cleaned_poem_classification_train_data.csv')
test_data = pd.read_csv('./poem_classification-test_data.csv')

# Data Cleaning
train_data['Genre'] = train_data['Genre'].str.strip()
train_data['Poem'] = train_data['Poem'].str.strip()
train_data = train_data.dropna()

test_data['Genre'] = test_data['Genre'].str.strip()
test_data['Poem'] = test_data['Poem'].str.strip()
test_data = test_data.dropna()

# Encode labels
le = LabelEncoder()
train_data['label'] = le.fit_transform(train_data['Genre'])
test_data['label'] = le.transform(test_data['Genre'])

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.eval()

MAX_LEN = 128

def encode_poem(poem):
    encoding = tokenizer.encode_plus(
        poem,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    return encoding

def get_bert_embeddings(poems):
    embeddings = []
    for poem in poems:
        encoding = encode_poem(poem)
        with torch.no_grad():
            outputs = bert_model(**encoding)
        cls_embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy()
        embeddings.append(cls_embedding)
    return np.vstack(embeddings)

# Get embeddings
X_train = get_bert_embeddings(train_data['Poem'].tolist())
y_train = np.array(train_data['label'])

X_test = get_bert_embeddings(test_data['Poem'].tolist())
y_test = np.array(test_data['label'])

# Stratified K-Fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
accuracies = []

for train_idx, val_idx in skf.split(X_train, y_train):
    print(f"\nFold {fold}")
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_tr, y_tr)
    y_val_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_val_pred)
    accuracies.append(acc)

    print("Validation Accuracy:", acc)
    print("Classification Report (Validation):\n", classification_report(y_val, y_val_pred))
    fold += 1

# Train final model on full training set
model.fit(X_train, y_train)

# Test Evaluation
y_test_pred = model.predict(X_test)
print("\nTest Accuracy:", accuracy_score(y_test, y_test_pred))
print("Classification Report (Test):\n", classification_report(y_test, y_test_pred))

def predict_genre(poem):
    encoding = encode_poem(poem)
    with torch.no_grad():
        outputs = bert_model(**encoding)
    poem_embedding = outputs.last_hidden_state.mean(dim=1).numpy().reshape(1, -1)
    genre_prediction = model.predict(poem_embedding)
    return le.inverse_transform(genre_prediction)[0]

# code updated "used stratifiedfold"