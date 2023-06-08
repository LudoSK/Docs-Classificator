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


# Function to manage the "Start" button
def start_classification():
    # Get selected model
    selected_model = model_var.get()

    # Get typed text
    input_text = text_entry.get("1.0", tk.END).strip()
    print(len(input_text))
    if not selected_model or not input_text:
        tk.messagebox.showerror("Text Classification","Model or input text is empty")
        return
    input_text = preprocess_text(input_text)
    input_text = ' '.join(input_text)



    vectorizer_name = selected_model.split("-")[0]
    model = load_model(selected_model)

    # Perform the classification and display the result
    if vectorizer_name == "w2v":
        model.vectorizer.train([input_text], total_examples=model.vectorizer.corpus_count, epochs=model.vectorizer.epochs)
        input_vectors = document_vector(model.vectorizer,input_text)
        input_vectors = csr_matrix(input_vectors)
    else:
        input_vectors = model.vectorizer.transform([input_text])

    result = model.classifier.predict(input_vectors)

    print(result)
    result_label.configure(text=f"Result : {result[0]}")


# Creation of the main window
window = tk.Tk()
window.title("Classification de texte")

# List of available models
models = ["BoW-MinDistance", "tfidf-MinDistance", "w2v-MinDistance", "BoW-Naive", "tfidf-Naive",
          "BoW-Logistic", "tfidf-Logistic", "w2v-Logistic"]

# Creation of the "Select Model" label
select_model_label = tk.Label(window, text="Select Model")
select_model_label.pack()

# Creating the model selector
model_var = tk.StringVar()
model_var.set(models[0])  # Default selection
model_selector = tk.OptionMenu(window, model_var, *models)
model_selector.pack()

# Creating the text box to enter the text
text_entry_label = tk.Label(window, text="Enter Text")
text_entry_label.pack()
text_entry = tk.Text(window, height=5, width=30)
text_entry.pack()

# Creation of the "Result" label
result_label = tk.Label(window, text="Résultat :")
result_label.pack()

# Creation of the "Start" button
start_button = tk.Button(window, text="Start", command=start_classification)
start_button.pack()

# Launching the main loop of the interface
window.mainloop()
