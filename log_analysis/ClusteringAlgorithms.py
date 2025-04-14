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

def KMeansClustering(X, n_clusters):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    return labels

def DBSCANclustering(X):
    '''
    eps: max distance for two points to be considered neighbors
    min_samples: min points to form a cluster
    label == -1 means noise (outlier) 
    '''
    from sklearn.cluster import DBSCAN
    db = DBSCAN(eps=0.5, min_samples=2)  # tune these params
    labels = db.fit_predict(X)
    return labels

def HDBSCANclustering(X):
    '''
    Automatically finds clusters, deals well with noisy or small data.
    '''
    import hdbscan 
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
    labels = clusterer.fit_predict(X)
    return labels

def AggloClustering(X):
    '''
    You can use scipy to draw a dendrogram too
    '''
    from sklearn.cluster import AgglomerativeClustering
    agg = AgglomerativeClustering(n_clusters=3)
    labels = agg.fit_predict(X.toarray())  # needs dense input
    return labels

def SpectrClustering(X):
    from sklearn.cluster import SpectralClustering
    spectral = SpectralClustering(n_clusters=3, affinity='nearest_neighbors')
    labels = spectral.fit_predict(X)
    return labels

def AffinityPropClustering(X):
    from sklearn.cluster import AffinityPropagation
    aff = AffinityPropagation(damping=0.8)
    labels = aff.fit_predict(X.toarray())
    return labels

if __name__ == "__main__":
    fail_logs = parse_xml("reports/output.xml")
    documents = [stringify_test_case(t) for t in fail_logs]

    # Embedding
    X = TF_IDFembedding(documents)

    # KMEANS
    kmeans_labels = KMeansClustering(X, n_clusters=3)

    # DBSCAN
    db_labels = DBSCANclustering(X)

    # HDBSCAN
    hdb_labels = HDBSCANclustering(X)

    # Agglomerative
    agg_labels = AggloClustering(X)

    # Spectral
    spec_labels = SpectrClustering(X)
