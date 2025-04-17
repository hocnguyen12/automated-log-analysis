
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
import tempfile

from XMLlogsParser import parse_xml
from JSONconverter import save_converted_xml_to_json
from Utils import build_log_text

# --- Paths ---
original_data_path = "dataset/train_fails.json"
new_data_path = "dataset/new_fails.json"
feedback_data_path = "dataset/feedback_fails.json"
model_path = "model/latest_classifier.pkl"
vectorizer_path = "model/latest_vectorizer.pkl"
faiss_index_path = "model/latest_faiss.index"

# Load existing model if available
if os.path.exists(model_path) and os.path.exists(vectorizer_path):
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

# --- Streamlit UI ---
st.title('Robot Framework Automated Tests Fail Analysis')

# Step 1: File Upload for new test log (output.xml) and multiple files upload
uploaded_files = st.file_uploader("Upload the new Robot Framework test execution (output.xml) files", type=["xml"], accept_multiple_files=True)

new_data = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Save the uploaded file to a temporary file on disk
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Parse the uploaded XML file (implement your XML parsing here)
        new_data.extend(save_converted_xml_to_json(tmp_file_path, new_data_path))

    st.write(f"{len(new_data)} failures parsed from uploaded files.")

    # Display the new fails
    for idx, fail in enumerate(new_data):
        st.subheader(f"Fail {idx+1}")
        st.write(f"Test Name: {fail['test_name']}")
        st.write(f"Error: {fail['error_message']}")
        st.write(f"Doc: {fail.get('doc', 'No description available.')}")
        if 'steps' in fail:
            for step in fail['steps']:
                st.write(f"Step: {step['keyword']}")
                st.write(f"Args: {' '.join(step['args'])}")
                st.write(f"Status: {step['status']}")
                st.write(f"Doc: {step.get('doc', 'No description available.')}")
                st.write(f"Messages: {' | '.join(step.get('messages', []))}")
        
        # Step 3: Add a "Predict Fix" button for each fail
        if st.button(f"Predict Fix for Fail {idx+1}", key=f"predict_{idx+1}"):
            # Prediction
            log_text = build_log_text(fail)
            new_vec = vectorizer.transform([log_text]) if vectorizer else None
            if clf and new_vec is not None:
                pred = clf.predict(new_vec)
                st.write(f"Predicted Fix Category: {pred[0]}")

            # Step 4: User Feedback (Was the suggested fix correct?)
            feedback = st.radio(f"Was the suggested fix correct for fail {idx+1}?", options=["Yes", "No"], key=f"feedback_{idx+1}")
            actual_category = None
            if feedback == "No":
                actual_category = st.text_input(f"Correct fix category for fail {idx+1}:", key=f"actual_category_{idx+1}")
            
            # Button for feedback submission
            if st.button(f"Submit Feedback for Fail {idx+1}", key=f"submit_{idx+1}"):
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
                    retrain_model_and_faiss(new_data)
                    st.write("Model and FAISS index have been updated.")

# Step 4: Button to retrain the model with all uploaded logs
if st.button("Train New Model with Uploaded Data"):
    st.write("Training new model and updating FAISS index...")
    retrain_model_and_faiss(new_data)
    st.write("Model and FAISS index have been updated.")

# --- Define helper functions ---

# Function to build log text from structured data
def build_log_text(item):
    msg = f"Test name: {item['test_name']}\n"
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
    print("Model and FAISS index have been updated.")
