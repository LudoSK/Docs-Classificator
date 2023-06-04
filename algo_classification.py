from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import np
from sklearn.feature_extraction.text import CountVectorizer
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
logistic_regression_classifier = LogisticRegression()

# Now, these classifiers can be trained using the fit method and the predictions can be made using the predict method.
# For example, to train the Minimum Distance Classifier:
# min_distance_classifier.fit(X_train, y_train)

# And to make predictions:
# y_pred = min_distance_classifier.predict(X_test)

# And to calculate the accuracy:
# accuracy = accuracy_score(y_test, y_pred)

def save_model(model, vectorizer_name, classifier_name):
    model_name = vectorizer_name + '-' + classifier_name + '.pkl'
    with open(model_name, 'wb') as file:
        pickle.dump(model, file)
    print("Modèle sauvegardé avec succès dans le fichier :", model_name)



# case 1 BoW & MinDistance
min_distance_classifier.fit(bow.toarray(), [item[1] for item in preprocessed_training_data])
out = count_vectorizer.transform([item[0] for item in preprocessed_testing_data])
predictions = min_distance_classifier.predict(out.toarray())
save_model(min_distance_classifier,"BoW", "MinDistance")

'''
# case 2 tfidf & MinDistance
min_distance_classifier.fit(tfvector.toarray(), [item[1] for item in preprocessed_training_data])
out2 = tfidf_vectorizer.transform([item[0] for item in preprocessed_testing_data])
predictions2 = min_distance_classifier.predict(out2.toarray())
'''
# case 3 w2v & MinDistance
min_distance_classifier.fit(w2v_vectors, [item[1] for item in preprocessed_training_data])
# Initialisez une liste pour stocker les vecteurs de mots du document
out3 = [document_vector(document) for document in [item[0] for item in preprocessed_testing_data]]

predictions3 = min_distance_classifier.predict(out3)
print(predictions3)


'''
# case 4 bow & Naive
naive_bayes_classifier.fit(bow.toarray(), [item[1] for item in preprocessed_training_data])
out4 = count_vectorizer.transform([item[0] for item in preprocessed_testing_data])
predictions4 = naive_bayes_classifier.predict(out4.toarray())
print(predictions4)


# case 5 tfidf & Naive
naive_bayes_classifier.fit(tfvector.toarray(), [item[1] for item in preprocessed_training_data])
out5 = tfidf_vectorizer.transform([item[0] for item in preprocessed_testing_data])
predictions5 = naive_bayes_classifier.predict(out5.toarray())
print(predictions5)

# case 6 w2v & Naive
#ça ne marche pas car les nombres sont négatifs


# case 7 bow & Logistic Regression
logistic_regression_classifier.fit(bow.toarray(), [item[1] for item in preprocessed_training_data])
out7 = count_vectorizer.transform([item[0] for item in preprocessed_testing_data])
predictions7 = logistic_regression_classifier.predict(out7.toarray())
print(predictions7)

# case 8 tfidf & Logistic Regression
logistic_regression_classifier.fit(tfvector.toarray(), [item[1] for item in preprocessed_training_data])
out8 = tfidf_vectorizer.transform([item[0] for item in preprocessed_testing_data])
predictions8 = logistic_regression_classifier.predict(out8.toarray())
print(predictions8)
'''


