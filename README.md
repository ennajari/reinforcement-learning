# Projets d'Apprentissage par Renforcement

<div align="center">
  <a href="https://github.com/ennajari/reinforcement-learning">
    <img src="https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=github" alt="GitHub">
  </a>
</div>

## Table des Matières
<!-- HTML version for better control -->
<div class="toc">
  <ul>
    <li><a href="#tp1">TP1: Découverte de Gymnasium</a></li>
    <li><a href="#tp2">TP2: Q-Learning</a></li>
    <li><a href="#tp3">TP3: Gestion de Trafic</a></li>
    <li><a href="#results">Résultats Clés</a></li>
    <li><a href="#install">Installation</a></li>
  </ul>
</div>

<h2 id="tp1">TP1: Découverte de Gymnasium et CartPole</h2>

### 🎯 Objectifs
```html
<div class="objectives">
  <ul>
    <li>Prise en main de Gymnasium</li>
    <li>Exploration de CartPole-v1</li>
    <li>Compréhension des concepts RL de base</li>
  </ul>
</div>
```
## 📝 Code Clé
    # Création de l'environnement
    env = gym.make("CartPole-v1", render_mode="human")
    observation, info = env.reset()

    # Boucle d'interaction
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
<h2 id="tp2">TP2: Q-Learning sur FrozenLake</h2>

## 🧠 Algorithme Principal
# Initialisation Q-Table
q_table = np.zeros((num_states, num_actions))

# Mise à jour Q-Learning
q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])

## 📊 Résultats
<div class="results">
  <table>
    <tr>
      <th>Métrique</th>
      <th>Valeur</th>
    </tr>
    <tr>
      <td>Taux de réussite</td>
      <td>75%</td>
    </tr>
    <tr>
      <td>Épisodes d'entraînement</td>
      <td>5000</td>
    </tr>
  </table>
</div>

<h2 id="tp3">TP3: Gestion de Trafic</h2>

## 🔄 Comparaison Algorithmes

<div style="text-align:center">
  <img src="https://github.com/ennajari/reinforcement-learning/raw/main/images/curves.png" width="500" alt="Courbes d'apprentissage">
</div>

## 📈 Tableau Comparatif

<table>
  <tr>
    <th>Algorithme</th>
    <th>Récompense Moyenne</th>
    <th>Stabilité</th>
  </tr>
  <tr>
    <td>Q-Learning</td>
    <td>42.7</td>
    <td>⭐⭐⭐⭐</td>
  </tr>
  <tr>
    <td>SARSA</td>
    <td>39.1</td>
    <td>⭐⭐⭐⭐⭐</td>
  </tr>
</table>

<h2 id="install">Installation</h2>

# Commandes d'installation
git clone https://github.com/ennajari/reinforcement-learning.git <br>
cd reinforcement-learning <br>
pip install -r requirements.txt <br>

<h2 id="results">Résultats Clés</h2>

<div class="highlight">
  <p>🔍 <strong>Découverte majeure</strong> : Q-Learning converge plus vite mais SARSA est plus stable dans cet environnement.</p>
</div>

<style> .toc ul { list-style-type: none; padding-left: 0; } .highlight { background-color: #f8f8f8; padding: 10px; border-radius: 5px; } table { border-collapse: collapse; width: 100%; margin: 20px 0; } th, td { border: 1px solid #ddd; padding: 8px; text-align: left; } th { background-color: #f2f2f2; } </style>




