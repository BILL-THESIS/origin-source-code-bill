from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

# Assuming you already have 'corpus' and 'text_df' prepared
tfidf_v = TfidfVectorizer(max_df=0.5, max_features=13000, min_df=5, stop_words='english', use_idf=True, norm='l2', smooth_idf=True)
X = tfidf_v.fit_transform(corpus)  # Removed .toarray() - not needed for KMeans
y = text_df.iloc[:, 1].values

# Initialize KMeans with 7 clusters
km = KMeans(n_clusters=7, random_state=42)
model = km.fit(X)
result = model.predict(X)

# Print some random predictions
for _ in range(20):
    j = np.random.randint(low=0, high=13833, size=1)[0]
    print(f"Author {y[j]} wrote {X[j]} and was put in cluster {result[j]}")
