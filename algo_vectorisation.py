import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

from text_preprocessing import preprocessed_training_data

# Bag of Word algorithm
# Initialize the CountVectorizer
count_vectorizer = CountVectorizer()

# Apply the CountVectorizer to your data
bow = count_vectorizer.fit_transform([item[0] for item in preprocessed_training_data])

# bow is now a sparse matrix that represents your documents


# TF-IDF algorithm
# Initialize the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Apply the TfidfVectorizer to your data
tfvector = tfidf_vectorizer.fit_transform([item[0] for item in preprocessed_training_data])

# tfidf is now a sparse matrix that represents your documents

# Split your documents into sentences
documents = [item[0] for item in preprocessed_training_data]

# Word2Vec algorithm
# Initialize the Word2Vec
w2v = Word2Vec(documents, min_count=1)

# Learn the vector representations of words
w2v.train(documents, total_examples=w2v.corpus_count, epochs=w2v.epochs)


def document_vector(w2vec,document):
    vector_sum = sum(abs(w2vec.wv[word]) for word in document if word in w2vec.wv.key_to_index)
    return vector_sum / len(document) if vector_sum is not None else np.zeros(100)


w2v_vectors = [document_vector(w2v,document) for document in documents]
