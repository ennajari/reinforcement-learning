# TP7: Détection de fausses offres d'emploi avec apprentissage par renforcement

Ce projet démontre l'application de l'apprentissage par renforcement à un problème de classification textuelle: la détection de fausses offres d'emploi. Contrairement aux approches classiques de classification supervisée, nous modélisons ce problème comme un environnement d'apprentissage par renforcement où l'agent DQN apprend à distinguer les offres légitimes des frauduleuses.

## Structure du projet

```
TP7/
├── Data/                                  # Dossier des données
│   ├── fake_job_postings.csv             # Dataset original
│   ├── fake_job_dqn_model.pth            # Modèle DQN entraîné  
│   ├── initial_dqn_model.pth             # État initial du modèle
│   ├── train_env.pkl                     # Environnement d'entraînement
│   ├── test_env.pkl                      # Environnement de test
│   ├── vectorizer.pkl                    # Vectoriseur TF-IDF
│   ├── X_train.npy                       # Features d'entraînement
│   ├── X_test.npy                        # Features de test
│   ├── y_train.npy                       # Labels d'entraînement
│   └── y_test.npy                        # Labels de test

├── notebooks/                            # Notebooks Jupyter
│   ├── 1_data_exploration.ipynb          # Exploration des données
│   ├── 2_data_preprocessing.ipynb        # Prétraitement des données
│   ├── 3_environment_setup.ipynb         # Configuration de l'environnement RL
│   ├── 4_dqn_implementation.ipynb        # Implémentation du DQN
│   ├── 5_model_training.ipynb           # Entraînement du modèle
│   └── 6_evaluation.ipynb               # Évaluation des performances

├── app/                                 # Application web
│   └── ui.py                            # Interface utilisateur Streamlit

├── requirements.txt                      # Dépendances Python
├── .gitignore                           # Fichiers à ignorer par Git
└── README.md                            # Documentation du projet                        # Documentation du projet
```

## Présentation du projet

### Objectifs
- Modéliser un problème de classification textuelle comme un environnement RL
- Construire un agent DQN capable de distinguer les offres d'emploi authentiques des frauduleuses
- Développer une interface utilisateur pour tester le modèle en situation réelle

### Données
Le dataset `fake_job_postings.csv` contient des offres d'emploi légitimes et frauduleuses avec leurs descriptions. Les caractéristiques des offres frauduleuses peuvent inclure:
- Demandes d'informations personnelles ou bancaires
- Promesses de revenus irréalistes
- Sentiment d'urgence excessif
- Absence de prérequis professionnels

## Méthodologie

### 1. Préparation des données
- Nettoyage et prétraitement du texte
- Vectorisation en utilisant TF-IDF
- Division en ensembles d'entraînement et de test

### 2. Modélisation en environnement RL
- **État**: Vecteur TF-IDF d'une offre d'emploi
- **Actions**: Prédiction (0: authentique, 1: frauduleuse)
- **Récompenses**: +1 pour classification correcte, -1 sinon
- **Environnement**: Créé en héritant de `gym.Env`

### 3. Apprentissage avec Deep Q-Network
- Architecture: Réseau à 3 couches fully-connected
- Exploration avec politique epsilon-greedy
- Expérience replay pour stabiliser l'apprentissage
- Suivi des métriques: perte, epsilon, récompenses

### 4. Évaluation et test en situation réelle
- Évaluation sur ensemble de test
- Test sur nouvelles offres jamais vues
- Analyse des caractéristiques à risque

## Améliorations apportées au projet initial

### Dans le notebook d'entraînement
- Visualisation des mots les plus importants dans la vectorisation TF-IDF
- Initialisation améliorée des poids du réseau (Kaiming)
- Suivi des métriques d'entraînement avec visualisations
- Matrice de confusion pour une meilleure évaluation des performances
- Test sur plusieurs exemples avec différents niveaux de risque

### Dans l'interface utilisateur
- Interface interactive avec Streamlit
- Analyse des caractéristiques de risque avec radar chart
- Jauge de risque basée sur les Q-values
- Exemples prédéfinis pour démonstration immédiate
- Recommandations personnalisées selon le niveau de risque détecté

## Comment utiliser ce projet

### Installation des dépendances
```bash
pip install numpy pandas torch scikit-learn gymnasium matplotlib seaborn streamlit
```

### Entraînement du modèle
1. Placez le dataset dans le dossier `Data/`
2. Exécutez le notebook `notebooks/TP7.ipynb`
3. Le modèle entraîné et le vectoriseur seront sauvegardés dans `app/`

### Utilisation de l'interface
```bash
cd app
streamlit run app.py
```

L'interface web permet de:
- Entrer une nouvelle offre d'emploi pour analyse
- Visualiser la prédiction et le niveau de confiance
- Explorer les caractéristiques à risque
- Tester des exemples prédéfinis

## Idées d'extension

Voici quelques pistes pour étendre ce projet:
1. **Scraping d'offres réelles**: Collecter des offres depuis LinkedIn ou Indeed pour tester en conditions réelles
2. **OCR d'images d'offres**: Extraire le texte d'images d'offres d'emploi pour analyse
3. **Génération d'offres avec LLM**: Utiliser des modèles comme GPT pour générer des offres frauduleuses et tester la robustesse
4. **Déploiement cloud**: Déployer l'application sur une plateforme cloud comme Heroku
5. **API REST**: Créer une API pour intégrer la détection dans d'autres services

## Crédits et références
- Dataset: "Fake Job Posting Prediction" disponible sur Kaggle
- Gymnasium: Framework pour la création d'environnements RL
- PyTorch: Framework pour l'implémentation du réseau de neurones
- Streamlit: Bibliothèque pour la création d'interfaces web#
