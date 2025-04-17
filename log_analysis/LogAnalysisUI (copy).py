import streamlit as st
import json
import os
import joblib
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import tempfile
from pathlib import Path

from Utils import parse_xml, build_log_text, auto_label_fix_category, merge_xml_training_data, save_converted_xml_to_json, BOLD, END

############################################# PATHS ####################################################
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
    st.write("Existing model loaded")
else:
    clf = None
    vectorizer = None
    st.write("No existing model found")

# Load FAISS index if available
if os.path.exists(faiss_index_path):
    faiss_index = faiss.read_index(faiss_index_path)
    st.write("FAISS index loaded")
else:
    faiss_index = None
    st.write("No FAISS index found")

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

########################################## HELPER FUNCTIONS ##################################################
def train_model():
    st.write("Loading data from Robot Framework test report...")
    with open(original_data_path, "r") as f:
        base_data = json.load(f)

    st.write("Loading data from User Feedback...")
    feedback_entries = []
    if Path(feedback_data_path).exists():
        with open(feedback_data_path, "r") as f:
            feedback_entries = [json.loads(line) for line in f if line.strip()]

    st.write("Labeling data (fail -> fix)...")
    base_data = auto_label_fix_category(base_data)

    merged = []

    st.write("preparing training data (x=fail, y=fix_category)...")
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

    #st.write("Training Data : ")
    #for entry in merged:
    #    print(json.dumps(entry, indent=1))

    texts = [r["log_text"] for r in merged]
    labels = [r["fix_category"] for r in merged]

    load = False

    st.write("No existing model found, training model...")
    vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
    X = vectorizer.fit_transform(texts)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    st.write("**Classification Report:**")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.write(report_df)

    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    st.write("Building FAISS index (similarity retrieval)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_index.add(np.array(embeddings))
    faiss.write_index(faiss_index, str(faiss_index_path))

    st.write("**Classification model and FAISS index saved to `log_analysis/model`**")

def predict(fail, clf, vectorizer):
    log_text = build_log_text(fail)
    new_vec = vectorizer.transform([log_text]) if vectorizer else None

    # Predict fix
    if clf and new_vec is not None:
        pred = clf.predict(new_vec)
        return pred[0], log_text
    return None, log_text

def show_similar_fails(log_text, faiss_index, model):
    query_embedding = model.encode([log_text], normalize_embeddings=True)
    D, I = faiss_index.search(np.array(query_embedding), k=3)
    
    similar_fails = []
    for rank, idx in enumerate(I[0]):
        similar_fails.append({
            "rank": rank + 1,
            "log": texts[idx][:300],  # Snippet
            "fix_category": labels[idx],
            "similarity_score": D[0][rank]
        })
    return similar_fails

##### SECTIONS
def prediction_section(fail, clf, vectorizer):
    if st.button(f"Predict Fix for Fail {fail['test_name']}"):
        pred, log_text = predict_fix(fail, clf, vectorizer)
        if pred:
            st.write(f"Predicted Fix Category: {pred}")
            st.session_state.predicted_fix = pred
            st.session_state.log_text = log_text
        else:
            st.write("Prediction failed!")

def similarity_section():
    if "log_text" in st.session_state:
        similar_fails = show_similar_fails(st.session_state.log_text, faiss_index, model)
        st.write("Top 3 Similar Fails:")
        for fail in similar_fails:
            st.write(f"Rank: {fail['rank']}, Similarity Score: {fail['similarity_score']}")
            st.write(f"Log Snippet: {fail['log']}")
            st.write(f"Fix Category: {fail['fix_category']}")
        
def feedback_section(fail):
    feedback = st.radio("Was the suggested fix correct?", options=["Yes", "No"])
    actual_category = None
    if feedback == "No":
        actual_category = st.text_input("What is the correct fix category?")

    if st.button(f"Submit Feedback for Fail {fail['test_name']}"):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "log_text": st.session_state.log_text,
            "predicted_category": st.session_state.predicted_fix if feedback == "Yes" else None,
            "feedback": feedback,
            "actual_category": actual_category if feedback == "No" else None
        }
        with open(feedback_data_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        
        st.write("Feedback submitted. Thank you!")


########################################### STREAMLIT UI ###############################################
st.title('Robot Framework Automated Tests Fail Analysis')

uploaded_files_train = st.file_uploader("**Upload past test execution as training data (optionally multiple output.xml files)**", type=["xml"],  accept_multiple_files=True)
##### TRAINING #####
if uploaded_files_train:
    ##st.write(f"UPLOADED FILES : {uploaded_files_train}")
    list_files = []
    for uploaded_file in uploaded_files_train:
        #st.write(f"PROCESSING FILE : {uploaded_file}")
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:  # 'wb' for binary mode
            tmp_file.write(uploaded_file.read()) 
            tmp_file_path = tmp_file.name
        list_files.append(tmp_file_path)

    #st.write(f"FILE LIST : {list_files}")
    merge_xml_training_data(list_files, original_data_path)
    if st.button(f"Train model", key=f"train_model"):
        train_model()

##### PREDICTION #####
uploaded_file_predict = st.file_uploader("Upload new test execution (output.xml)", type=["xml"])
if uploaded_file_predict is not None:
    with tempfile.NamedTemporaryFile(delete=False, mode='wb') as tmp_file:
        tmp_file.write(uploaded_file_predict.read())  # Save the uploaded file
        tmp_file_path = tmp_file.name

    # Process and display prediction and similar fails
    fail = parse_xml(tmp_file_path)[0]  # Assume single test fail
    prediction_section(fail, clf, vectorizer)
    similarity_section()
    feedback_section(fail)

uploaded_file_predict = st.file_uploader("**Upload the new Robot Framework test execution (output.xml)**", type=["xml"])
if uploaded_file_predict:
    if clf and vectorizer and faiss_index:
        #st.write(uploaded_file_predict)
        # Save the uploaded file to a temporary file on disk
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file_predict.read())
            tmp_file_path = tmp_file.name

        #st.write(f"File saved to: {tmp_file_path}")

        new_data = save_converted_xml_to_json(tmp_file_path, new_data_path)

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

                #if clf and new_vec is not None:
                pred = clf.predict(new_vec)
                st.write(f"**[PREDICTION] Fail Correction: {pred[0]}**")
                    
                if st.button(f"Show most similar fails from database", key=f"similarity_{idx+1}"):
                    query_embedding = model.encode([log_text], normalize_embeddings=True)
                    D, I = faiss_index.search(np.array(query_embedding), k=3)

                    st.write("***Top 3 similar failures:**")
                    for rank, idx in enumerate(I[0]):
                        st.write(f"\n#{rank+1} (Score: {D[0][rank]:.2f})")
                        st.write("Log:", texts[idx][:300].replace("\n", " ") + "...")
                        st.write("Fix Category:", labels[idx])

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
    else :
        st.write("First train a model and FAISS index")
