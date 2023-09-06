import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

# Load data
df = pd.read_csv("pages/Research_Articles.csv")

# Reset index and rename columns
df = df.reset_index()
df = df.rename(columns={"index": "id"})

# Select a subset of the data (optional)
df1 = df.head(500)

# Fill NaN values
df1['ABSTRACT'] = df1['ABSTRACT'].fillna('')

# Load SentenceTransformer model
model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L6-cos-v5')

# Encode the corpus
corpus = df1['ABSTRACT'].tolist()
corpus_embeddings = model.encode(corpus)

# Calculate cosine similarity between sentence embeddings
cosine_similarities = cosine_similarity(corpus_embeddings)

# Perform hierarchical agglomerative clustering
num_clusters = 5
cluster_model = AgglomerativeClustering(n_clusters=num_clusters, affinity='precomputed', linkage='average')
cluster_assign = cluster_model.fit_predict(1 - cosine_similarities)

# Organize sentences into clusters
cluster_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assign):
    cluster_sentences[cluster_id].append(corpus[sentence_id])

# Streamlit app
st.title('Semantic Clustering for Medical Research Articles')

# Display clusters
for i, cluster in enumerate(cluster_sentences):
    st.header(f'Cluster {i + 1}')
    st.write(cluster)

