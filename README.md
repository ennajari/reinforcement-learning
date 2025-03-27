# Projets d'Apprentissage par Renforcement

<div align="center">
  <a href="https://github.com/ennajari/reinforcement-learning">
    <img src="https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=github" alt="GitHub">
  </a>
</div>

## Table des Matières

<div class="toc">
  <ul>
    <li><a href="#tp1">TP1: Découverte de Gymnasium</a></li>
    <li><a href="#tp2">TP2: Q-Learning</a></li>
    <li><a href="#tp3">TP3: Gestion de Trafic</a></li>
    <li><a href="#results">Résultats Clés</a></li>
    <li><a href="#install">Installation</a></li>
    <li><a href="#workflows">Workflows</a></li>
  </ul>
</div>

<h2 id="tp1">TP1: Découverte de Gymnasium et CartPole</h2>

### 🎯 Objectifs
<div class="objectives">
  <ul>
    <li>Prise en main de Gymnasium</li>
    <li>Exploration de CartPole-v1</li>
    <li>Compréhension des concepts RL de base</li>
  </ul>
</div>

### 📝 Code Clé
### Création de l'environnement
    import gymnasium as gym
    env = gym.make("CartPole-v1", render_mode="human")
    observation, info = env.reset()

### Boucle d'interaction
    for _ in range(100):
        action = env.action_space.sample()  # Action aléatoire
        observation, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            observation, info = env.reset()
        
### 🔄 Workflow CartPole

```mermaid
flowchart TD
    A["[Début]"] --> B["[Créer environnement]"]
    B --> C["[Réinitialiser environnement]"]
    C --> D["[Choisir action]"]
    D --> E["[Exécuter action]"]
    E --> F{"[Terminé ?]"}
    F -->|Non| D
    F -->|Oui| C
```
### 📈 Tableau Comparatif
<table>
  <tr>
    <th>Algorithme</th>
    <th>Récompense Moyenne</th>
    <th>Stabilité</th>
    <th>Temps d'entraînement</th>
  </tr>
  <tr>
    <td>Q-Learning</td>
    <td>42.7 ± 3.2</td>
    <td>⭐⭐⭐⭐</td>
    <td>3 min</td>
  </tr>
  <tr>
    <td>SARSA</td>
    <td>39.1 ± 2.8</td>
    <td>⭐⭐⭐⭐⭐</td>
    <td>5 min</td>
  </tr>
</table>

<h2 id="install">Installation</h2>

# Cloner le dépôt
    git clone https://github.com/ennajari/reinforcement-learning.git
    cd reinforcement-learning

# Installer les dépendances
    pip install -r requirements.txt

# Exécuter les TPs
     tp1.ipynb  # CartPole
     tp2.ipynb  # FrozenLake
     tp3.ipynb  # Traffic Management

<h2 id="results">Résultats Clés</h2>

<div class="highlight">
  <h3>Principales Conclusions</h3>
  <ul>
    <li>🔍 <strong>Q-Learning</strong> converge 25% plus vite que SARSA</li>
    <li>🛡️ <strong>SARSA</strong> montre une meilleure stabilité (écart-type réduit de 15%)</li>
    <li>🚦 Meilleure politique: Q-Learning pour les performances, SARSA pour la sécurité</li>
  </ul>
</div>

<h2 id="workflows">Workflows Complets</h2>

### Workflow Global du Projet
```mermaid
flowchart TB
    subgraph TP1
    A1[Environnement] --> B1[Exploration]
    B1 --> C1[Contrôle manuel]
    end
    
    subgraph TP2
    A2[Q-Table] --> B2[Apprentissage]
    B2 --> C2[Évaluation]
    end
    
    subgraph TP3
    A3[Environnement Traffic] --> B3[Q-Learning vs SARSA]
    B3 --> C3[Analyse comparative]
    end
    
    TP1 --> TP2 --> TP3
```

<style>
  .toc ul {
    list-style-type: none;
    padding-left: 0;
  }
  
  .highlight {
    background-color: #f8f8f8;
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #4285f4;
    margin: 15px 0;
  }

  table {
    border-collapse: collapse;
    width: 100%;
    margin: 20px 0;
    box-shadow: 0 2px 3px rgba(0,0,0,0.1);
  }

  th, td {
    border: 1px solid #ddd;
    padding: 12px;
    text-align: left;
  }

  th {
    background-color: #f2f2f2;
    position: sticky;
    top: 0;
  }

  tr:nth-child(even) {
    background-color: #f9f9f9;
  }

  tr:hover {
    background-color: #f1f1f1;
  }
</style>
