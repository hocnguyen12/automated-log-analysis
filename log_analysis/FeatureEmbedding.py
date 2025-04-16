from XMLlogsParser import extract_keywords, parse_xml, pretty_print_fails, stringify_test_case
# Clustering
from sklearn.cluster import KMeans
# Data visualisation
import pandas as pd

def TF_IDFembedding(data):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
    X = vectorizer.fit_transform(data) # Sparse Matrix (n_tests x n_features)
    return X

def sentence_embedding(data):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    X = model.encode(data)
    return X






if __name__ == "__main__":
    fail_logs = parse_xml("reports/output.xml")
    #pretty_print_fails(fail_logs)

    documents = [stringify_test_case(t) for t in fail_logs]
    #print(f"\nstringified test cases :\n {documents}")

    #X = TF_IDFembedding(documents)
    X = sentence_embedding(documents)

    # KMeans Clustering
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)

    # Visualising Results
    results = pd.DataFrame({
        "name": [t["name"] for t in fail_logs],
        "error": [t["error_message"] for t in fail_logs],
        "cluster": labels
    })

    print(results.sort_values(by="cluster"))
