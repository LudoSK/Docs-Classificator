import os.path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import copy
from text_preprocessing import preprocessed_training_data
from text_preprocessing import preprocessed_testing_data
from algo_vectorisation import bow, tfidf_vectorizer
from algo_vectorisation import count_vectorizer
from algo_vectorisation import tfvector
from algo_vectorisation import w2v
from algo_vectorisation import w2v_vectors
from algo_vectorisation import document_vector
from sklearn.metrics import accuracy_score
import pickle

# Classifier 1: Minimum Distance Classifier
min_distance_classifier = KNeighborsClassifier(n_neighbors=1)

# Classifier 2: Naive Bayes Classifier
naive_bayes_classifier = MultinomialNB()

# Classifier 3: Logistic Regression
logistic_regression_classifier = LogisticRegression(max_iter=200000)


# Now, these classifiers can be trained using the fit method and the predictions can be made using the predict method.
# For example, to train the Minimum Distance Classifier:
# min_distance_classifier.fit(X_train, y_train)

# And to make predictions:
# y_pred = min_distance_classifier.predict(X_test)

# And to calculate the accuracy:
# accuracy = accuracy_score(y_test, y_pred)

class Model:
    def __init__(self, vectorizer, classifier):
        self.vectorizer = vectorizer
        self.classifier = classifier


def save_model(vectorizer, classifier, vectorizer_name, classifier_name):
    model = Model(vectorizer, classifier)
    model_name = vectorizer_name + '-' + classifier_name + '.pkl'
    with open('models/' + model_name, 'wb') as file:
        pickle.dump(model, file)
    print("Model successfully saved to file :", model_name)


def train_and_save_model(vectorizer, classifier, vectorizer_name, classifier_name):
    if vectorizer_name == "BoW":
        classifier.fit(bow.toarray(), [item[1] for item in preprocessed_training_data])
        out = vectorizer.transform([item[0] for item in preprocessed_testing_data])
        predictions = classifier.predict(out.toarray())
        save_model(vectorizer, classifier, vectorizer_name, classifier_name)

    elif vectorizer_name == "tfidf":
        classifier.fit(tfvector.toarray(), [item[1] for item in preprocessed_training_data])
        out = vectorizer.transform([item[0] for item in preprocessed_testing_data])
        predictions = classifier.predict(out.toarray())
        save_model(vectorizer, classifier, vectorizer_name, classifier_name)

    elif vectorizer_name == "w2v":
        classifier.fit(w2v_vectors, [item[1] for item in preprocessed_training_data])
        out = [document_vector(w2v, document) for document in [item[0] for item in preprocessed_testing_data]]
        predictions = classifier.predict(out)
        save_model(w2v, classifier, vectorizer_name, classifier_name)


    else:
        raise ValueError("Invalid vectorizer name")
    accuracy = accuracy_score([item[1] for item in preprocessed_testing_data],predictions)

    return accuracy

# case 1 BoW & MinDistance
if not os.path.exists('models/BoW-MinDistance.pkl'):
    accuracy = train_and_save_model(copy.copy(count_vectorizer), copy.copy(min_distance_classifier), "BoW", "MinDistance")
    print(f"""Model 1 : BoW-MinDistance
            Accuracy : {accuracy*100}%""")


# case 2 tfidf & MinDistance
if not os.path.exists('models/tfidf-MinDistance.pkl'):
    accuracy = train_and_save_model(copy.copy(tfidf_vectorizer), copy.copy(min_distance_classifier), "tfidf", "MinDistance")
    print(f"""Model 2 : tfidf-MinDistance
                Accuracy : {accuracy * 100}%""")

# case 3 w2v & MinDistance
if not os.path.exists('models/w2v-MinDistance.pkl'):
    accuracy = train_and_save_model(copy.copy(w2v), copy.copy(min_distance_classifier), "w2v", "MinDistance")
    print(f"""Model 3 : w2v-MinDistance
                    Accuracy : {accuracy * 100}%""")

# case 4 bow & Naive
if not os.path.exists('models/BoW-Naive.pkl'):
    accuracy = train_and_save_model(copy.copy(count_vectorizer), copy.copy(naive_bayes_classifier), "BoW", "Naive")
    print(f"""Model 4 : models/BoW-Naive
                    Accuracy : {accuracy * 100}%""")

# case 5 tfidf & Naive
if not os.path.exists('models/tfidf-Naive.pkl'):
    accuracy = train_and_save_model(copy.copy(tfidf_vectorizer), copy.copy(naive_bayes_classifier), "tfidf", "Naive")
    print(f"""Model 5 : tfidf-Naive
                    Accuracy : {accuracy * 100}%""")

# case 6 w2v & Naive
#it doesn't work because the numbers from w2v are negative

# case 7 bow & Logistic Regression
if not os.path.exists('models/BoW-Logistic.pkl'):
    accuracy = train_and_save_model(copy.copy(count_vectorizer), copy.copy(logistic_regression_classifier), "BoW", "Logistic")
    print(f"""Model 7 : BoW-Logistic
                    Accuracy : {accuracy * 100}%""")

# case 8 tfidf & Logistic Regression
if not os.path.exists('models/tfidf-Logistic.pkl'):
    accuracy = train_and_save_model(copy.copy(tfidf_vectorizer), copy.copy(logistic_regression_classifier), "tfidf", "Logistic")
    print(f"""Model 8 : tfidf-Logistic
                    Accuracy : {accuracy * 100}%""")

# case 9 w2v & Logistic Regression
if not os.path.exists('models/w2v-Logistic.pkl'):
    accuracy = train_and_save_model(copy.copy(w2v), copy.copy(logistic_regression_classifier), "w2v", "Logistic")
    print(f"""Model 9 : w2v-Logistic
                    Accuracy : {accuracy * 100}%""")































