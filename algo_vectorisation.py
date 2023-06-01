from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

from text_preprocessing import preprocessed_training_data

# Bag of Word algorithm
# Initialize the CountVectorizer
vectorizer = CountVectorizer()

# Apply the CountVectorizer to your data
bow = vectorizer.fit_transform(preprocessed_training_data)

# bow is now a sparse matrix that represents your documents


# TF-IDF algorithm
# Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer()

# Apply the TfidfVectorizer to your data
tfidf = vectorizer.fit_transform(preprocessed_training_data)

# tfidf is now a sparse matrix that represents your documents

# Split your documents into sentences
sentences = [doc.split(' ') for doc in preprocessed_training_data]



# Word2Vec algorithm
# Initialize the Word2Vec
model = Word2Vec(sentences, min_count=1)

# Learn the vector representations of words
model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)

# You can now get the vector representation of a word with model.wv['word']
