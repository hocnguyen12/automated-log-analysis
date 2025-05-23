'''
Commands :
    pip install hdbscan

Table :
Need                              | Algo
----------------------------------|-------------------------------
Automatically find #clusters	  |  HDBSCAN, Affinity Propagation
Handle outliers (errors)	      |  DBSCAN, HDBSCAN
Hierarchical relationships	      |  Agglomerative
High quality semantic grouping	  |  Spectral, HDBSCAN
Fast and simple baseline	      |  KMeans
'''
from XMLlogsParser import extract_keywords, parse_xml, pretty_print_fails, stringify_test_case
from FeatureEmbedding import TF_IDFembedding, sentence_embedding
import pandas as pd
import scipy.sparse
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import hdbscan 
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AffinityPropagation

def KMeansClustering(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    return labels

def DBSCANclustering(X):
    '''
    eps: max distance for two points to be considered neighbors
    min_samples: min points to form a cluster
    label == -1 means noise (outlier) 
    '''
    db = DBSCAN(eps=0.5, min_samples=2)  # tune these params
    labels = db.fit_predict(X)
    return labels

def HDBSCANclustering(X):
    '''
    Automatically finds clusters, deals well with noisy or small data.
    '''
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
    labels = clusterer.fit_predict(X)
    return labels

def AggloClustering(X):
    '''
    You can use scipy to draw a dendrogram too
    '''
    if scipy.sparse.issparse(X):
        X = X.toarray()

    agg = AgglomerativeClustering(n_clusters=3)
    labels = agg.fit_predict(X)  # needs dense input
    return labels

def SpectrClustering(X):
    n_samples = X.shape[0]
    n_neighbors = min(n_samples - 1, 5)  # or any smaller number

    spectral = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', n_neighbors=n_neighbors)
    labels = spectral.fit_predict(X)
    return labels

def AffinityPropClustering(X):
    if scipy.sparse.issparse(X):
        X = X.toarray()
        
    aff = AffinityPropagation(damping=0.8)
    labels = aff.fit_predict(X)
    return labels




if __name__ == "__main__":
    fail_logs = parse_xml("reports/output.xml")
    documents = [stringify_test_case(t) for t in fail_logs]

    # Embedding
    #X = TF_IDFembedding(documents)
    X = sentence_embedding(documents)

    kmeans_labels = KMeansClustering(X, n_clusters=3) # KMEANS
    db_labels = DBSCANclustering(X) # DBSCAN
    hdb_labels = HDBSCANclustering(X) # HDBSCAN
    agg_labels = AggloClustering(X) # Agglomerative
    spec_labels = SpectrClustering(X) # Spectral

    # Clustering Visualisation
    results = pd.DataFrame({
        "name": [t["name"] for t in fail_logs],
        "error": [t["error_message"] for t in fail_logs],
        "cluster": kmeans_labels
    })
    df = pd.DataFrame({
        "name": [t["name"] for t in fail_logs],
        "error": [t["error_message"] for t in fail_logs],
        "kmeans": kmeans_labels,
        "dbscan": db_labels,
        "hdbscan": hdb_labels,
        "agglo": agg_labels,
        "spectral": spec_labels
    })
    #print(results.sort_values(by="cluster"))
    print(df)
