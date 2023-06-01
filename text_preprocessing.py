import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from download_process_data import data

# Download the necessary resources for NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Define the English stop words
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenize the text
    words = word_tokenize(text)

    # Remove stop words and lemmatize each word
    return [lemmatizer.lemmatize(word) for word in words if word not in stop_words and word.isalpha()]

# For each document in 'data'
for doc in data:
    # Preprocess the document's text
    preprocessed_text = preprocess_text(doc['text'])

    # Store the preprocessed text back in the dictionary
    doc['text'] = ' '.join(preprocessed_text)

# At this point, each document in 'data' has its text preprocessed.

preprocessed_training_data = [(doc['text'], doc['class']) for doc in data if doc['split'] == 'TRAIN']
preprocessed_testing_data = [(doc['text'], doc['class']) for doc in data if doc['split'] == 'TEST']
