"""
Interface utilisateur pour la détection de fausses offres d'emploi
TP7 - Apprentissage par Renforcement appliqué à la classification textuelle
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
from io import BytesIO
import base64
import os

# Vérification des fichiers nécessaires
REQUIRED_FILES = ['../Data/vectorizer.pkl', '../Data/fake_job_dqn_model.pth']

# Définition de la structure du modèle DQN
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Fonction de prétraitement du texte
def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

# Fonction pour évaluer une offre d'emploi
def evaluate_job(title, description, model, vectorizer):
    if not title or not description:
        return None, None, None
    
    text = clean_text(title + ' ' + description)
    features = vectorizer.transform([text]).toarray().astype(np.float32)
    
    with torch.no_grad():
        state_tensor = torch.FloatTensor(features[0]).unsqueeze(0)
        q_values = model(state_tensor)
        prediction = torch.argmax(q_values[0]).item()
        proba_fake = F.softmax(q_values, dim=1)[0][1].item()
    
    return prediction, q_values[0].numpy(), proba_fake

# Fonction pour générer un graphique des q-values
def plot_q_values(q_values):
    if q_values is None:
        return None
        
    fig, ax = plt.subplots(figsize=(10, 6))
    classes = ['Authentique', 'Frauduleuse']
    colors = ['#2ecc71', '#e74c3c'] if q_values[0] > q_values[1] else ['#e74c3c', '#2ecc71']
    
    ax.bar(classes, q_values, color=colors)
    ax.set_title('Q-values pour cette offre d\'emploi', fontsize=15)
    ax.set_ylabel('Q-value', fontsize=12)
    
    for i, v in enumerate(q_values):
        ax.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=12)
    
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    confidence = abs(q_values[0] - q_values[1])
    ax.text(0.5, max(q_values) * 1.1, f'Différence: {confidence:.4f}', 
             ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    return fig

# Fonction pour générer un radar chart des caractéristiques de risque
def plot_risk_features(title, description):
    if not title or not description:
        return None
        
    # Caractéristiques de risque à évaluer
    risk_features = {
        'Urgence': len(re.findall(r'urgent|immediate|asap|quick', title.lower() + ' ' + description.lower())),
        'Argent facile': len(re.findall(r'easy money|rich|wealth|income|profit|earn \$', description.lower())),
        'Infos perso': len(re.findall(r'bank details|personal info|credit card|ssn|passport', description.lower())),
        'Télétravail': len(re.findall(r'work from home|remote|wfh', title.lower() + ' ' + description.lower())),
        'Promesses': len(re.findall(r'guarantee|promise|best opportunity|lifetime', description.lower()))
    }
    
    # Normalisation des scores (entre 0 et 5)
    max_val = max(risk_features.values()) if max(risk_features.values()) > 0 else 1
    risk_features = {k: min(5, v * (5/max_val)) for k, v in risk_features.items()}
    
    # Préparation des données pour le radar chart
    labels = list(risk_features.keys())
    values = list(risk_features.values())
    
    # Fermer le cercle en ajoutant le premier point à la fin
    values += values[:1]
    
    # Angles pour chaque axe
    N = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Fermer le cercle
    
    # Création du radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Dessiner le polygone
    ax.fill(angles, values, color='#3498db', alpha=0.25)
    ax.plot(angles, values, color='#3498db', linewidth=2)
    
    # Ajouter les étiquettes
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    
    # Ajouter les valeurs sur chaque axe
    for angle, value, label in zip(angles[:-1], values[:-1], labels):
        ax.text(angle, value + 0.3, f"{value:.1f}", ha='center', va='center')
    
    plt.title('Analyse des caractéristiques à risque', size=15, y=1.1)
    plt.tight_layout()
    
    return fig

# Interface utilisateur avec Streamlit
def main():
    st.set_page_config(
        page_title="Détecteur d'offres d'emploi frauduleuses",
        page_icon="",
        layout="wide"
    )
    
    st.title("Détecteur d'offres d'emploi frauduleuses")
    st.markdown("""
    Cette application utilise l'apprentissage par renforcement pour détecter les offres d'emploi frauduleuses. 
    Entrez le titre et la description d'une offre pour l'analyser.
    """)
    
    # Vérification des fichiers nécessaires
    missing_files = [f for f in REQUIRED_FILES if not os.path.exists(f)]
    if missing_files:
        st.error(f"Fichiers manquants: {', '.join(missing_files)}. Veuillez placer ces fichiers dans le même répertoire que l'application.")
        return
    
    # Chargement du modèle et du vectoriseur
    @st.cache_resource
    def load_model():
        try:
            with open('../Data/vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
            
            input_dim = 5000
            model = DQN(input_dim, 2)
            model.load_state_dict(torch.load('../Data/fake_job_dqn_model.pth', map_location=torch.device('cpu')))
            model.eval()
            
            return model, vectorizer
        except Exception as e:
            st.error(f"Erreur lors du chargement du modèle : {str(e)}")
            return None, None
    
    model, vectorizer = load_model()
    
    if model is None or vectorizer is None:
        return
    
    # Onglets
    tab1, tab2, tab3 = st.tabs(["Analyse d'offre", "Exemples prédéfinis", "À propos"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Entrez les détails de l'offre d'emploi")
            job_title = st.text_input("Titre de l'offre", placeholder="Ex: Data Scientist - Remote Work")
            job_description = st.text_area("Description de l'offre", 
                                          height=200, 
                                          placeholder="Entrez la description complète de l'offre...")
            
            analyze_button = st.button("Analyser l'offre", type="primary")
            
        with col2:
            st.subheader("Résultat de l'analyse")
            
            if analyze_button:
                if not job_title or not job_description:
                    st.warning("Veuillez entrer un titre et une description pour analyser l'offre.")
                else:
                    with st.spinner("Analyse en cours..."):
                        prediction, q_values, proba_fake = evaluate_job(job_title, job_description, model, vectorizer)
                        
                        if prediction == 1:
                            st.error("ATTENTION: Cette offre est potentiellement FRAUDULEUSE!")
                        else:
                            st.success("Cette offre semble AUTHENTIQUE.")
                        
                        st.markdown("### Niveau de risque")
                        st.progress(proba_fake)
                        st.write(f"Probabilité de fraude: {proba_fake:.2%}")
                        
                        q_fig = plot_q_values(q_values)
                        if q_fig:
                            st.pyplot(q_fig)
                        
                        st.markdown("### Analyse des facteurs de risque")
                        risk_fig = plot_risk_features(job_title, job_description)
                        if risk_fig:
                            st.pyplot(risk_fig)
                        
                        st.markdown("### Recommandations")
                        if prediction == 1:
                            st.warning("""
                            - Méfiez-vous des demandes d'informations personnelles ou financières
                            - Recherchez l'entreprise en ligne avant de postuler
                            - Vérifiez si l'entreprise a un site web professionnel
                            - Ne payez jamais de frais pour postuler à un emploi
                            """)
                        else:
                            st.info("""
                            - Cette offre semble légitime, mais restez vigilant
                            - Vérifiez l'entreprise sur des sites d'avis comme Glassdoor
                            - Ne communiquez jamais d'informations sensibles prématurément
                            """)
            else:
                st.info("Entrez le titre et la description d'une offre d'emploi, puis cliquez sur 'Analyser l'offre'.")
    
    with tab2:
        st.subheader("Exemples d'offres d'emploi")
        st.write("Cliquez sur un exemple pour l'analyser instantanément.")
        
        examples = [
            {
                "title": "Software Developer - Junior Position",
                "description": "We're looking for a passionate junior developer to join our team. Requirements: Basic knowledge of programming languages like Python or JavaScript. We offer mentorship, competitive salary, and a friendly work environment. Send your resume to our official HR email.",
                "type": "Authentique"
            },
            {
                "title": "URGENT - Make Money Fast - Work From Home!!!",
                "description": "Make $5000 weekly working just 2 hours a day! No experience needed! Just send us your bank details to get started immediately. Limited positions available! Act now before it's too late! 100% guaranteed income!",
                "type": "Frauduleuse"
            }
        ]
        
        for i, example in enumerate(examples):
            col1, col2, col3 = st.columns([3, 6, 1])
            with col1:
                st.write(f"**{example['title']}**")
            with col2:
                st.write(example['description'][:100] + "...")
            with col3:
                if st.button(f"Analyser", key=f"example_{i}"):
                    with st.spinner("Analyse en cours..."):
                        prediction, q_values, proba_fake = evaluate_job(example['title'], example['description'], model, vectorizer)
                        
                        st.markdown("---")
                        st.subheader(f"Analyse de l'exemple: {example['title']}")
                        st.write(f"**Description complète:** {example['description']}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if prediction == 1:
                                st.error("FRAUDULEUSE")
                            else:
                                st.success("AUTHENTIQUE")
                            
                            st.markdown(f"**Étiquette réelle:** {example['type']}")
                            st.markdown(f"**Probabilité de fraude:** {proba_fake:.2%}")
                        
                        with col2:
                            q_fig = plot_q_values(q_values)
                            if q_fig:
                                st.pyplot(q_fig)
                        
                        risk_fig = plot_risk_features(example['title'], example['description'])
                        if risk_fig:
                            st.pyplot(risk_fig)
                        st.markdown("---")
    
    with tab3:
        st.subheader("À propos de cette application")
        
        st.markdown("""
        ### Contexte du projet
        Cette application a été développée pour démontrer l'utilisation de l'apprentissage par renforcement 
        pour la classification de texte. Le modèle a été entraîné pour distinguer les offres d'emploi 
        authentiques des frauduleuses.
        
        ### Caractéristiques techniques
        - **Modèle**: Deep Q-Network (DQN)
        - **Vectorisation**: TF-IDF avec 5000 features
        - **Framework**: PyTorch
        - **Interface**: Streamlit
        
        ### Signaux d'alerte
        Les offres frauduleuses présentent souvent :
        - Promesses de revenus élevés pour peu d'effort
        - Demandes d'informations personnelles/bancaires
        - Sentiment d'urgence exagéré
        - Manque de détails sur l'entreprise
        """)
        
        st.info("""
        **Note**: Cette application est une démonstration. Dans un contexte réel, 
        des vérifications supplémentaires seraient nécessaires pour une détection fiable.
        """)

if __name__ == "__main__":
    main()