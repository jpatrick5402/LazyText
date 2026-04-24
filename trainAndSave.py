import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib

# --- Data ---
# Labels: 0 = No Response, 1 = AI Response, 2 = Human Response
df = pd.read_csv('training_data.csv', quotechar='"', escapechar='\\')
texts = df['text'].tolist()
labels = df['label'].tolist()

# --- Embed ---
embedder = SentenceTransformer('all-MiniLM-L6-v2')
X = embedder.encode(texts, show_progress_bar=False)

# --- Cross-validate ---
model = SVC(kernel='rbf', probability=True, C=1.0)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_f1 = cross_val_score(model, X, labels, cv=cv, scoring='f1_weighted')
scores_acc = cross_val_score(model, X, labels, cv=cv, scoring='accuracy')
print(f"F1:       {scores_f1.mean():.2f} ± {scores_f1.std():.2f}")
print(f"Accuracy: {scores_acc.mean():.2f} ± {scores_acc.std():.2f}")

# --- Train on all data ---
model.fit(X, labels)

# --- Classify ---
def classify_message(text):
    vec = embedder.encode([text])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][pred]
    labels_map = {0: "No Response Needed", 1: "AI Response", 2: "Human Response"}
    return pred, f"{labels_map[pred]} ({prob:.0%} confidence)"

# --- Save ---
joblib.dump(model, 'message_classifier.pkl')

# --- Test ---
test_messages = [
    "Hey!",
    "I love you son.",
    "Can you call me?",
    "We're having a baby!",
    "Ok sounds good",
    "I just got diagnosed with cancer.",
    "Did you see the game last night?",
    "Want to grab dinner this week?",
    "lol",
    "We need to talk.",
    "What is your name?",
    "Who am I",
    "I'm so bored",
    "I farted on a pickle",
    "I miss you",
    "Hey son, do you know why Richard isn't texting me?"
]

print("\n--- Test Results ---")
for msg in test_messages:
    _, result = classify_message(msg)
    print(f"'{msg}' → {result}")
