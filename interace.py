import tkinter as tk
from tkinter import messagebox
import pickle

def load_model(vectorizer_name, classifier_name):
    model_name = vectorizer_name + '-' + classifier_name + '.pkl'
    with open(model_name, 'rb') as file:
        model = pickle.load(file)
    return model


# Fonction pour gérer le bouton "Start"
def start_classification():
    # Obtenir le modèle sélectionné
    selected_model = model_var.get()

    # Obtenir le texte saisi
    input_text = text_entry.get("1.0", tk.END).strip()

    # Effectuer la classification et afficher le résultat
    if selected_model == "Model 1":
        result_label.config(text="Résultat : Modèle 1 - " + input_text)
    elif selected_model == "Model 2":
        result_label.config(text="Résultat : Modèle 2 - " + input_text)
    elif selected_model == "Model 3":
        result_label.config(text="Résultat : Modèle 3 - " + input_text)
    elif selected_model == "Model 4":
        result_label.config(text="Résultat : Modèle 4 - " + input_text)
    elif selected_model == "Model 5":
        result_label.config(text="Résultat : Modèle 5 - " + input_text)
    elif selected_model == "Model 6":
        result_label.config(text="Résultat : Modèle 6 - " + input_text)


# Création de la fenêtre principale
window = tk.Tk()
window.title("Classification de texte")

# Liste des modèles disponibles
models = ["Model 1", "Model 2", "Model 3", "Model 4", "Model 5", "Model 6"]

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
