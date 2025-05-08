# app.py

import streamlit as st
import torch
import pickle
import numpy as np
import re
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

# Nettoyage du texte
def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

# Définition du modèle DQN (même structure que dans ton TP)
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Chargement du vectorizer et du modèle
vectorizer_path = 'vectorizer.pkl'
model_path = 'fake_job_dqn_model.pth'

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

input_dim = 5000  # Même valeur que TfidfVectorizer(max_features=5000)
model = DQN(input_dim, 2)
model.load_state_dict(torch.load(model_path))
model.eval()

# Interface utilisateur Streamlit
st.title("🕵️ Détection de Fausse Offre d'Emploi (DQN)")

title = st.text_input("Titre de l'offre", "Data Scientist - Remote Work")
description = st.text_area("Description", "Exciting opportunity...")

if st.button("Analyser"):
    combined_text = clean_text(title + " " + description)
    features = vectorizer.transform([combined_text]).toarray().astype(np.float32)
    with torch.no_grad():
        state_tensor = torch.FloatTensor(features[0]).unsqueeze(0)
        q_values = model(state_tensor)
        prediction = torch.argmax(q_values[0]).item()
        classe = "🚨 FAUSSE" if prediction == 1 else "✅ RÉELLE"

    st.subheader(f"Résultat de la Prédiction : {classe}")
    st.write(f"Q-values : {q_values[0].numpy()}")

    # Affichage graphique
    st.subheader("Visualisation des Q-values")
    fig, ax = plt.subplots()
    ax.bar(['RÉELLE', 'FAUSSE'], q_values[0].numpy(), color=['green', 'red'])
    ax.set_ylabel("Q-value")
    st.pyplot(fig)
