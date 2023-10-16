#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

data = pd.read_json("C:\\Users\\mohra\\Downloads\\arxiv-metadata-oai-snapshot.json", lines=True, nrows=1000 )


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


import re

# Define a function to preprocess text
def preprocess_text(text):
    # Cleaning special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Apply text preprocessing to the 'abstract' column
data['abstract'] = data['abstract'].apply(preprocess_text)


# In[7]:


data


# In[8]:


#TF-IDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf_vectorizer.fit_transform(data['abstract'])


# In[9]:


from sklearn.cluster import KMeans

k = 5
kmeans = KMeans(n_clusters=k)
data['cluster'] = kmeans.fit_predict(tfidf_matrix)



# In[10]:


#Principal Component Analysis (PCA) for Dimensionality Reduction

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(tfidf_matrix.toarray())


# In[11]:


#visualitation : scatter plot cluster

import matplotlib.pyplot as plt

plt.scatter(pca_result[:,0],pca_result[:,1], c=data['cluster'], cmap='viridis') 
plt.title('PCA Scatter Plot of Clusters')
plt.show()


# In[12]:


#word cloud for clustering

from wordcloud import WordCloud

k = 5

for cluster in range(k):
    cluster_text = ' '.join(data[data['cluster']==cluster]['abstract'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cluster_text)
    plt.figure(figsize=(8,4))
    plt.imshow(wordcloud)
    plt.title(f'word cloud for clustering')
    plt.axis('off')
    plt.show()


# In[ ]:




