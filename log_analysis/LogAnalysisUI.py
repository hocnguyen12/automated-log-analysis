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
import streamlit_nested_layout
import matplotlib.pyplot as plt

from Utils import parse_xml, build_log_text, auto_label_fix_category, merge_xml_training_data, save_converted_xml_to_json, BOLD, END, stringify_test_case, sentence_embedding
from Utils import KMeansClustering, DBSCANclustering, HDBSCANclustering, AggloClustering, SpectrClustering, AffinityPropClustering

st.title('Robot Framework Automated Tests Fail Analysis')

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
    st.write("`Existing model loaded`")
else:
    clf = None
    vectorizer = None
    st.write("`No existing model found`")

# Load FAISS index if available
if os.path.exists(faiss_index_path):
    faiss_index = faiss.read_index(faiss_index_path)
    st.write("`FAISS index loaded`")
else:
    faiss_index = None
    st.write("`No FAISS index found`")

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
st.write("`Sentence transformer loaded`")

########################################## HELPER FUNCTIONS ##################################################
def merge_train_data(print=False):
    if print: st.write("`Loading data from Robot Framework test report...`")
    with open(original_data_path, "r") as f:
        base_data = json.load(f)

    if print: st.write("`Loading data from User Feedback...`")
    feedback_entries = []
    if Path(feedback_data_path).exists():
        with open(feedback_data_path, "r") as f:
            feedback_entries = [json.loads(line) for line in f if line.strip()]

    if print: st.write("`Labeling data (fail -> correction)...`")
    base_data = auto_label_fix_category(base_data)

    merged = []

    if print: st.write("`preparing training data (x=fail, y=fix_category)...`")
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
    return merged

def get_texts_and_labels(merged_fails):
    texts = [r["log_text"] for r in merged_fails]
    labels = [r["fix_category"] for r in merged_fails]
    return texts, labels

def train_model(texts, labels):
    st.write("`No existing model found, training model...`")
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

    st.write("`Building FAISS index (similarity retrieval)...`")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_index.add(np.array(embeddings))
    faiss.write_index(faiss_index, str(faiss_index_path))

    st.write("**Model saved to `log_analysis/model`**")


def train_model_new(texts, labels):
    st.write("`Loading existing model if available...`")
    
    # Load existing model and vectorizer
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        clf = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        st.write("`Existing model loaded, retraining...`")
    else:
        # If no model exists, create a new one
        clf = RandomForestClassifier()
        vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
        st.write("`No existing model found, training new model...`")
    
    # Transform the new texts using the vectorizer
    X_new = vectorizer.fit_transform(texts)
    
    # If model exists, combine old and new data for retraining
    if clf:
        # Assuming you are training the model again with old data
        if hasattr(clf, "fit"):  # Check if the classifier has the fit method
            # You could consider loading old data and combining it with the new data here
            X_old, y_old = load_existing_training_data()  # You should implement this function to load old data
            X_combined = np.vstack((X_old, X_new.toarray()))  # Combine old and new data
            y_combined = np.concatenate((y_old, labels))  # Combine old and new labels
            clf.fit(X_combined, y_combined)  # Retrain on the combined data
        else:
            # If the model doesn't support incremental training, this method would be invalid
            st.write("`Model does not support incremental training, re-training from scratch.`")
            clf.fit(X_new, labels)
    
    # Evaluate the retrained model
    y_pred = clf.predict(X_new)
    st.write("**Classification Report:**")
    report = classification_report(labels, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.write(report_df)
    
    # Save the retrained model and vectorizer
    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    # Build FAISS index for new data
    st.write("`Building or updating FAISS index (similarity retrieval)...`")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_index.add(np.array(embeddings))
    faiss.write_index(faiss_index, str(faiss_index_path))
    
    st.write("**Model and FAISS index saved to `log_analysis/model`**")


def get_similar_fails(log_text, faiss_index, model):
    query_embedding = model.encode([log_text], normalize_embeddings=True)
    D, I = faiss_index.search(np.array(query_embedding), k=3)

    merged = merge_train_data()
    texts, labels = get_texts_and_labels(merged)
    
    similar_fails = []
    for rank, idx in enumerate(I[0]):
        similar_fails.append({
            "rank": rank + 1,
            #"log": texts[idx][:300],  # Snippet
            "log": texts[idx], 
            "fix_category": labels[idx],
            "similarity_score": D[0][rank]
        })
    return similar_fails

def dislay_fail_in_expander(fail):
    st.write(
        f"""
        ### Test Case: {fail['test_name']}
        Error: `{fail['error_message']}`

        Doc: {fail.get('doc', 'No description available')}
        """
    )
    if 'steps' in fail:
        for step in fail['steps']:
            with st.expander(f"**Step:** {step['keyword']}"):
                st.write(f"- **Args**: `{' '.join(step['args'])}`")
                st.write(f"- **Status**: `{step['status']}`")
                st.write(f"- **Doc**: {step.get('doc', 'No description available.')}")
                st.write(f"- **Messages**: `{' | '.join(step.get('messages', []))}`")

##############################################  SECTIONS ##################################################
def prediction_section(fail, idx, clf, vectorizer):
    # Predict Button
    if f"predicted_fix_{idx+1}" not in st.session_state:
        if st.button(f"Predict fail correction", key=f"predict_{idx+1}"):
            #st.session_state[f"predict_button_{idx+1}"] = not st.session_state[f"predict_button_{idx+1}"]
            # Perform prediction
            log_text = build_log_text(fail)
            new_vec = vectorizer.transform([log_text]) if vectorizer else None
            pred = clf.predict(new_vec)
            st.write(f"**[Suggested Correction]: {pred[0]}**")
            st.session_state[f"predicted_fix_{idx+1}"]= pred[0]
            st.session_state[f"log_text_{idx+1}"] = log_text 
    else :
        if st.button(f"Predict fail correction", key=f"predict_{idx+1}"):
            pass
        state = f"predicted_fix_{idx+1}"
        st.write(f"**[Suggested Correction]: {st.session_state[state]}**")

def similarity_section(idx, recall=True):
    if f"log_text_{idx+1}" in st.session_state and f"predicted_fix_{idx+1}" in st.session_state:
        if st.button("Show Most Similar Fails", key=f"similarity_button_{idx+1}"):
            similar_fails = get_similar_fails(st.session_state[f"log_text_{idx+1}"], faiss_index, model)
            st.write("**Top 3 Similar Fails:**")
            for fail in similar_fails:
                with st.expander(f"- **Rank: {fail['rank']}** [Similarity Score = {fail['similarity_score']}]"):
                    st.write(f"Log: `{fail['log']}`")
                    st.write(f"**Correction**: {fail['fix_category']}")
            # When page is rerun, similar fails will still be showed
            st.session_state[f"show_similar_fails_{idx+1}"] = True
        elif f"show_similar_fails_{idx+1}" in st.session_state and st.session_state[f"show_similar_fails_{idx+1}"]:
            similar_fails = get_similar_fails(st.session_state[f"log_text_{idx+1}"], faiss_index, model)
            st.write("**Top 3 Similar Fails:**")
            for fail in similar_fails:
                with st.expander(f"- **Rank: {fail['rank']}** [Similarity Score = {fail['similarity_score']}]"):
                    st.write(f"Log Snippet: `{fail['log']}`")
                    st.write(f"**Correction**: {fail['fix_category']}")


def feedback_section(fail, idx):
    if f"feedback_{idx+1}" not in st.session_state:
        st.session_state[f"feedback_{idx+1}"] = False

    if f"predicted_fix_{idx+1}" in st.session_state:
        if st.button("Give Feedback", key=f"feedback_button_{idx+1}"):
            st.session_state[f"feedback_{idx+1}"] = not st.session_state[f"feedback_{idx+1}"]

            st.write("Is the suggested correction right ?")

        if st.session_state[f"feedback_{idx+1}"]:
            if st.button("Yes", key=f"yes_button_{idx+1}"):
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "log_text": st.session_state[f"log_text_{idx+1}"],
                    "predicted_category": st.session_state[f"predicted_fix_{idx+1}"],
                    "feedback": "correct",
                    "actual_category": st.session_state[f"predicted_fix_{idx+1}"]
                }
                with open(feedback_data_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
                #write_feedback(entry)
                st.write("Feedback saved. Thank you!")
                st.session_state[f"feedback_{idx+1}"] = False

        if st.session_state[f"feedback_{idx+1}"]:
            if st.button("No", key=f"no_button_{idx+1}"):
                # If the feedback is 'No', ask for the correct fix category
                actual_category = st.text_input("What is the right fix category?", key=f"actual_category_input_{idx+1}")
                
                if actual_category:
                    st.session_state[f"actual_category_input_{idx+1}"] = actual_category

            if f"actual_category_input_{idx+1}" in st.session_state and st.session_state[f"actual_category_input_{idx+1}"]:

                st.write("You entered: ", st.session_state[f"actual_category_input_{idx+1}"])
                # Save the feedback with the corrected fix category
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "log_text": st.session_state[f"log_text_{idx+1}"],
                    "predicted_category": st.session_state[f"predicted_fix_{idx+1}"],
                    "feedback": "wrong",
                    "actual_category": st.session_state[f"actual_category_input_{idx+1}"]  # Save the user-provided actual fix
                }
                with open(feedback_data_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
                #write_feedback(entry)
                st.write("Feedback saved. Thank you!")
                st.session_state[f"feedback_{idx+1}"] = False

########################################### STREAMLIT UI ###############################################
tab_predict, tab_train, tab_clustering = st.tabs(["Analyse fails", "Train model", "Clustering"])

with tab_train:
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
            merged = merge_train_data(print=True)
            texts, labels = get_texts_and_labels(merged)
            train_model(texts, labels)

##### PREDICTION #####
with tab_predict:
    uploaded_file_predict = st.file_uploader("**Upload the new Robot Framework test execution (output.xml)**", type=["xml"])
    if uploaded_file_predict:
        if clf and vectorizer and faiss_index:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file_predict.read())
                tmp_file_path = tmp_file.name

            new_data = save_converted_xml_to_json(tmp_file_path, new_data_path) # Save json file `dataset/new_fails.json`

            # Display the new fails
            for idx, fail in enumerate(new_data):
                with st.expander(f"Fail {idx+1} - {fail['test_name']}"):
                    dislay_fail_in_expander(fail) # Display Fail

                    prediction_section(fail, idx, clf, vectorizer) # Predict Button

                    similarity_section(idx) # FAISS Similarity Button

                    feedback_section(fail, idx) # Feedback Buttons (Correct / Incorrect)
        else :
            st.write("**First train a model and FAISS index**")


with tab_clustering:
    uploaded_file_cluster = st.file_uploader("**Upload Robot Framework test execution (output.xml)**", type=["xml"])
    if uploaded_file_cluster:     
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file_cluster.read())
            tmp_file_path = tmp_file.name

        fail_logs = parse_xml(tmp_file_path)
        documents = [stringify_test_case(t) for t in fail_logs]
        # Embedding
        X = sentence_embedding(documents)

        # User selects the clustering algorithm
        algorithm = st.selectbox("Choose Clustering Algorithm", ["KMeans", "DBSCAN", "HDBSCAN", "Agglomerative", "Spectral", "Affinity Propagation"])

        # Apply the selected clustering algorithm
        if algorithm == "KMeans":
            n_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)
            cluster_labels = KMeansClustering(X, n_clusters)
        elif algorithm == "DBSCAN":
            cluster_labels = DBSCANclustering(X)
        elif algorithm == "HDBSCAN":
            cluster_labels = HDBSCANclustering(X)
        elif algorithm == "Agglomerative":
            cluster_labels = AggloClustering(X)
        elif algorithm == "Spectral":
            cluster_labels = SpectrClustering(X)
        elif algorithm == "Affinity Propagation":
            cluster_labels = AffinityPropClustering(X)

        # Create a DataFrame with results
        results_df = pd.DataFrame({
            "name": [t["test_name"] for t in fail_logs],
            "error": [t["error_message"] for t in fail_logs],
            "cluster_label": cluster_labels
        })

        # Display the results
        st.write("Clustering Results:")
        st.dataframe(results_df)

        # Optionally, you can also plot the results if the data is 2D or you want to reduce dimensions
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(X)

            st.write("Cluster Visualization (PCA reduced):")
            #scatter = st.pyplot()
            
            fig, ax = plt.subplots()
            ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap='viridis')
            ax.set_title(f"Clustering: {algorithm}")
            st.pyplot(fig)

        except Exception as e:
            st.write(f"Error visualizing clusters: {e}")