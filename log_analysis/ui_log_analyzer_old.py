import streamlit as st
import json
import os
import joblib
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

from XMLlogsParser import parse_xml
from JSONconverter import save_converted_xml_to_json

# --- Paths ---
original_data_path = "structured_failures.json"
feedback_data_path = "feedback_log.json"
model_path = "model/latest_classifier.pkl"
vectorizer_path = "model/latest_vectorizer.pkl"
faiss_index_path = "model/latest_faiss.index"

# Load existing model if available
load = False
if os.path.exists(model_path) and os.path.exists(vectorizer_path) and load == True:
    clf = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
else:
    clf = None
    vectorizer = None

# Load FAISS index if available
if os.path.exists(faiss_index_path):
    faiss_index = faiss.read_index(faiss_index_path)
else:
    faiss_index = None

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Streamlit UI ---:
st.title('Robot Framework Automated Tests Fail Analysis')

# Step 1: File Upload for new test log (output.xml)
uploaded_file = st.file_uploader("Upload the new Robot Framework test output (output.xml)", type=["xml"])

if uploaded_file is not None:
    # Parse the uploaded XML file (implement your XML parsing here)
    new_data = parse_xml(uploaded_file)

    json_data = [convert_to_json_structured(t) for t in fail_logs]
    print(f"JSON data : {json_data}")
    with open("log_analysis/analized_fails.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    st.write("New Test Fail Log:")
    st.json(new_data)  # Display the log (or a snippet of it)

    # Step 2: Perform Prediction
    log_text = build_log_text(new_data)  # Build log text (you can reuse your existing function here)
    new_vec = vectorizer.transform([log_text]) if vectorizer else None
    
    if clf and new_vec is not None:
        pred = clf.predict(new_vec)
        st.write(f"Predicted Fix Category: {pred[0]}")  # Show prediction

    # Step 3: FAISS - Retrieve similar past failures
    query_embedding = model.encode([log_text], normalize_embeddings=True)
    D, I = faiss_index.search(np.array(query_embedding), k=3)

    st.write("Top 3 Similar Past Failures:")
    for rank, idx in enumerate(I[0]):
        st.write(f"#{rank+1} (Score: {D[0][rank]:.2f})")
        st.write(f"Log: {texts[idx][:300]}...")
        st.write(f"Fix Category: {labels[idx]}")

    # Step 4: User Feedback (Was the suggested fix correct?)
    feedback = st.radio("Was the suggested fix correct?", options=["Yes", "No"])
    actual_category = None
    if feedback == "No":
        actual_category = st.text_input("What is the correct fix category?", "")
    
    if st.button("Submit Feedback"):
        # Save feedback
        entry = {
            "timestamp": datetime.now().isoformat(),
            "log_text": log_text,
            "predicted_category": pred[0] if feedback == "Yes" else None,
            "feedback": feedback,
            "actual_category": actual_category if feedback == "No" else None
        }

        with open(feedback_data_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        
        st.write("Feedback submitted. Thank you!")
        
        # Optionally retrain the model after feedback
        retrain = st.checkbox("Retrain model after feedback?")
        if retrain:
            # Append new data, retrain the model, and update the FAISS index
            retrain_model_and_faiss(merged_data)
            st.write("Model and FAISS index have been updated.")

# --- Define helper functions ---

# Function to build log text from structured data
def build_log_text(item):
    msg = f"Test name: {item['test_name']}\n"
    msg += f"Doc: {item.get('doc', '')}\n"
    msg += f"Error: {item['error']}\n"
    for step in item.get("steps", []):
        msg += f"Step: {step['keyword']}\n"
        msg += f"Args: {' '.join(step['args'])}\n"
        msg += f"Status: {step['status']}\n"
        if step.get("doc"):
            msg += f"Doc: {step['doc']}\n"
        if step.get("messages"):
            msg += f"Messages: {' | '.join(step['messages'])}\n"
    return msg

# Function to retrain the model and update the FAISS index
def retrain_model_and_faiss(merged_data):
    # Update the model and FAISS index with the new feedback data
    texts = [r["log_text"] for r in merged_data]
    labels = [r["fix_category"] for r in merged_data]
    
    vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
    X = vectorizer.fit_transform(texts)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    
    # Save the updated model and vectorizer
    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    # Rebuild FAISS index
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_index.add(np.array(embeddings))
    
    faiss.write_index(faiss_index, faiss_index_path)
