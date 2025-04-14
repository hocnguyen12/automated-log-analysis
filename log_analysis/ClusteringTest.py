from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd

# Extract only the error messages
error_messages = [entry["message"] for entry in failed_tests]

# Convert messages to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(error_messages)

# Apply KMeans clustering (choose 2 clusters for this example)
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X)

# Organize results into a DataFrame for easier viewing
clustered_errors = pd.DataFrame({
    "Test Name": [entry["name"] for entry in failed_tests],
    "Error Message": error_messages,
    "Cluster": labels
})

clustered_errors.sort_values(by="Cluster")
