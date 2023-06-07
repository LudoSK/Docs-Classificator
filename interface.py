import tkinter as tk
from tkinter import messagebox
import pickle
from algo_vectorisation import document_vector
from text_preprocessing import preprocess_text
from scipy.sparse import csr_matrix
from algo_classification import Model



def load_model(model_name):
    with open('models/'+model_name+ '.pkl', 'rb') as file:
        model = pickle.load(file)
    return model


# Fonction pour gérer le bouton "Start"
def start_classification():
    # Obtenir le modèle sélectionné
    selected_model = model_var.get()

    # Obtenir le texte saisi
    input_text = text_entry.get("1.0", tk.END).strip()
    print(len(input_text))
    if not selected_model or not input_text:
        tk.messagebox.showerror("Classification de texte","Model or input text is empty")
        return
    input_text = preprocess_text(input_text)
    input_text = ' '.join(input_text)



    vectorizer_name = selected_model.split("-")[0]
    model = load_model(selected_model)

    # Effectuer la classification et afficher le résultat
    if vectorizer_name == "w2v":
        model.vectorizer.train([input_text], total_examples=model.vectorizer.corpus_count, epochs=model.vectorizer.epochs)
        input_vectors = document_vector(model.vectorizer,input_text)
        input_vectors = csr_matrix(input_vectors)
    else:
        input_vectors = model.vectorizer.transform([input_text])

    result = model.classifier.predict(input_vectors)

    print(result)
    result_label.configure(text=f"Result : {result[0]}")


# Création de la fenêtre principale
window = tk.Tk()
window.title("Classification de texte")

# Liste des modèles disponibles
models = ["BoW-MinDistance", "tfidf-MinDistance", "w2v-MinDistance", "BoW-Naive", "tfidf-Naive",
          "BoW-Logistic", "tfidf-Logistic", "w2v-Logistic"]

# Création du label "Select Model"
select_model_label = tk.Label(window, text="Select Model")
select_model_label.pack()

# Création du sélecteur de modèle
model_var = tk.StringVar()
model_var.set(models[0])  # Sélection par défaut
model_selector = tk.OptionMenu(window, model_var, *models)
model_selector.pack()

# Création de la zone de texte pour saisir le texte
text_entry_label = tk.Label(window, text="Enter Text")
text_entry_label.pack()
text_entry = tk.Text(window, height=5, width=30)
text_entry.pack()

# Création du label "Résultat"
result_label = tk.Label(window, text="Résultat :")
result_label.pack()

# Création du bouton "Start"
start_button = tk.Button(window, text="Start", command=start_classification)
start_button.pack()

# Lancement de la boucle principale de l'interface
window.mainloop()
