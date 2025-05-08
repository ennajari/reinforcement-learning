graph TD
    A[agent] --> B[Effectue une action]
    B --> C[L'environnement réagit]
    C --> D[Modèle M = Règles de l’environnement]
    D --> E[Permet à l'agent de prédire les conséquences]
    E --> F[Anticiper et planifier les actions]
    G[L'agent sans modèle] --> H[Doit apprendre par essai-erreur]
    H --> I[Apprentissage plus lent]
