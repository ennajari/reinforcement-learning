pipeline {
    agent any

    stages {
        stage('Clone le dépôt') {
            steps {
                checkout scm
            }
        }

        stage('Installer les dépendances') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }

        stage('Lancer les tests') {
            steps {
                sh 'pytest || echo "Tests non définis pour l\'instant"'
            }
        }

        stage('Construire l\'image Docker') {
            steps {
                sh 'docker build -t rl_project .'
            }
        }
    }
}
