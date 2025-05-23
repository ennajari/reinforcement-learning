# Utilise une image Python officielle avec support de Jupyter
FROM python:3.12-slim

# Définit le répertoire de travail
WORKDIR /app

# Copie les fichiers nécessaires
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copie tout le reste du projet
COPY . .

# Expose le port pour Jupyter Notebook
EXPOSE 8888

# Démarre Jupyter Notebook automatiquement
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
