print("Loading libraries...")
import json
import os
import pandas as pd
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from datetime import datetime


# --- Config Paths ---
original_data_path = Path("structured_failures.json")
feedback_data_path = Path("feedback_log.json")
model_path = Path("model/latest_classifier.pkl")
vectorizer_path = Path("model/latest_vectorizer.pkl")
faiss_index_path = Path("model/latest_faiss.index")

BOLD = '\033[1m'
END = '\033[0m'

print(BOLD + "Loading data from Robot Framework test report..." + END)
with open(original_data_path, "r") as f:
    base_data = json.load(f)

print(BOLD + "Loading data from User Feedback..." + END)
feedback_entries = []
if feedback_data_path.exists():
    with open(feedback_data_path, "r") as f:
        feedback_entries = [json.loads(line) for line in f if line.strip()]

def auto_label_fix_category(data):
    for item in data:
        if "fix_category" not in item or not item["fix_category"]:
            error = item["error_message"].lower()
            if "missing" in error and "argument" in error:
                item["fix_category"] = "missing_argument"
            elif "not found" in error or "selector" in error:
                item["fix_category"] = "invalid_selector"
            elif "assert" in error or "should be equal" in error:
                item["fix_category"] = "assertion_failed"
            elif "timeout" in error:
                item["fix_category"] = "timeout"
            elif "connection" in error:
                item["fix_category"] = "connection_error"
            else:
                item["fix_category"] = "other"
    return data

print(BOLD + "Labeling data (fail -> fix)..." + END)
base_data = auto_label_fix_category(base_data)

merged = []

def build_log_text(item):
    msg = f"Test name: {item['test_name']}\n"
    msg += f"Doc: {item.get('doc', '')}\n"
    msg += f"Error: {item['error_message']}\n"
    for step in item.get("steps", []):
        msg += f"Step: {step['keyword']}\n"
        msg += f"Args: {' '.join(step['args'])}\n"
        msg += f"Status: {step['status']}\n"
        if step.get("doc"):
            msg += f"Doc: {step['doc']}\n"
        if step.get("messages"):
            msg += f"Messages: {' | '.join(step['messages'])}\n"
    return msg

print(BOLD + "preparing training data (x=fail, y=fix_category)..." + END)
for item in base_data:
    merged.append({
        "log_text": build_log_text(item),
        "fix_category": item["fix_category"]
    })

# For tests that have positive feedback OR correction/actual fix informed by the user
# -> add the test to training data
for fb in feedback_entries:
    if fb.get("feedback") == "correct":
        fix_category = fb.get("predicted_category", "unknown") # If the feedback is correct, use the predicted category 
    elif fb.get("feedback") == "wrong":
        fix_category = fb.get("actual_category", "unknown")  # If the feedback is wrong, use the actual category from the user
    else:
        fix_category = "unknown"

    merged.append({
        "log_text": fb["log_text"],
        "fix_category": fix_category
    })

# Filter out "unknown", "null" and "" fix categories before training
merged = [entry for entry in merged if entry["fix_category"] not in ["unknown", "null", ""]]

print(BOLD + "Training Data : " + END)
for entry in merged:
    print(json.dumps(entry, indent=1))

texts = [r["log_text"] for r in merged]
labels = [r["fix_category"] for r in merged]

load = False

if model_path.exists() and vectorizer_path.exists() and load == True:
    print(BOLD + "Existing model found, loading model..." + END)
    clf = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
else:
    print(BOLD + "No existing model found, training model..." + END)
    vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
    X = vectorizer.fit_transform(texts)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(BOLD + "\nClassification Report:\n" + END, classification_report(y_test, y_pred))
    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vectorizer_path)

print(BOLD + "Building FAISS index (similarity retrieval)..." + END)
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
faiss_index.add(np.array(embeddings))
faiss.write_index(faiss_index, str(faiss_index_path))

new_data = {
    "test_name": "Connect without API key",
    "error_message": "TypeError: TestObject.__init__() missing 1 required positional argument: 'api_key'",
    "doc": "Attempts to connect to the server without providing API key.",
    "steps": [
        {
            "keyword": "Connect",
            "args": ["http://localhost"],
            "status": "FAIL",
            "doc": "Connects to backend server using TestObject",
            "messages": ["Connecting to http://localhost", "Exception raised: missing 'api_key'"]
        }
    ]
}

log_text = build_log_text(new_data)

print(BOLD + "Predicting fix for new fail..." + END)
new_vec = vectorizer.transform([log_text])
pred = clf.predict(new_vec)
print(BOLD + "\n[PREDICTION] Suggested Fix Category:" + END, pred[0])

print(BOLD +"Looking for similar fails..." + END)
query_embedding = model.encode([log_text], normalize_embeddings=True)
D, I = faiss_index.search(np.array(query_embedding), k=3)

print(BOLD + "\nTop 3 similar failures:" + END)
for rank, idx in enumerate(I[0]):
    print(f"\n#{rank+1} (Score: {D[0][rank]:.2f})")
    print("Log:", texts[idx][:300].replace("\n", " ") + "...")
    print("Fix Category:", labels[idx])

feedback = input(BOLD + "\nWas the suggested fix correct? (y/n): " + END)

actual = None
if feedback.lower() == "n":  # If the feedback is 'no', ask for the correct fix category
    actual = input("What is the correct fix category? (type and press Enter): ")
    if actual is None:
        actual = "unknown"

# Output user feedback
if feedback.lower() == "y":
    print("User confirmed the fix was correct.")
else:
    print(f"User said the fix was incorrect. The correct fix category is: {actual}")

entry = {
    "timestamp": datetime.now().isoformat(),
    "log_text": log_text,
    "predicted_category": pred[0],
    "feedback": "correct" if feedback.lower() == "y" else "wrong",
    "actual_category": pred[0] if feedback.lower() == "y" else actual
}

print(BOLD + "Saving feedback for future predictions..." + END)
with open(feedback_data_path, "a") as f:
    f.write(json.dumps(entry) + "\n")

print(BOLD + "\n✅ Feedback saved." + END)