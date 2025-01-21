# Pole_Scraping: Un Projet de Web Scraping et Machine Learning

Bienvenue dans le projet **Pole_Scraping**, une application combinant les techniques de web scraping et de machine learning pour analyser et prédire les performances en qualifications de Formule 1. Ce projet vise à extraire, nettoyer, modéliser et prédire des données liées aux qualifications des pilotes et des équipes.

---

## Pourquoi Pole Scraping ?
Dans le cadre de notre cours intitulé « Web Scraping et Machine Learning » de M2 MECEN à l'Université de Tours, nous avons choisi de réaliser un projet visant à prédire une grille de départ en Formule 1 (résultats des qualifications).
Cette idée nous est venue grâce à un jeu que nous avons créé au sein de notre groupe d'amis, tous fans de Formule 1. Ce jeu consiste à prédire la grille de départ du prochain Grand Prix en plaçant un maximum de pilotes à la bonne position. Déterminés à maximiser nos chances de victoire, nous avons décidé d'utiliser tous les outils à notre disposition pour améliorer nos prédictions.
Nous avons opté pour les qualifications car elles sont confrontées à beaucoup moins d'aléas que la course. Les problèmes mécaniques y sont moins fréquents, et les accidents entre pilotes sont rares, car les qualifications ne reposent pas sur une confrontation directe entre eux.

## Organisation du projet
Le premier objectif de ce projet a été de constituer une base de données. Pour cela, nous avons utilisé la méthode de web scraping, qui consiste à extraire des données directement depuis des sites internet afin de créer une base de données personnalisée. Cette dernière regroupe diverses informations sur les pilotes, les écuries et les circuits.

La seconde partie de ce projet repose sur l'utilisation du Machine Learning pour réaliser les prédictions. Bien que notre objectif soit de prédire les positions des pilotes, nous n'avons pas retenu la variable "position", car il s'agit d'une variable discrète et les modèles que nous avons essayé prédisent une variable continue. Nous avons donc opté pour la variable "temps_sec" (correspondant au meilleur temps au tour réalisé par un pilote lors des qualifications), à laquelle nous avons appliqué une modification.
Pour adapter cette variable, nous avons effectué une normalisation contextuelle en fonction de la longueur du circuit. Plus précisément, nous avons créé une nouvelle variable, "Allure", en divisant le temps en secondes par la longueur du circuit (en kilomètres) : temps_sec / longueur_km. Ainsi, notre modèle se concentre sur la prédiction de cette variable "Allure".
Une fois l'allure prédite pour chaque Pilote, il suffit de trier les résultats obtenus dans l'odre croissant de la variable allure et nous obtenons une grille de Grand Prix.

## Fonctionnalités Principales
- **Web Scraping** : Collecte automatisée de données depuis des sources fiables.
- **Nettoyage de Données** : Prétraitement et organisation des données pour une meilleure qualité d'analyse.
- **Prédiction Machine Learning** : Modèle de boosting pour prédire les allures des pilotes.
- **Visualisation Intéractive** : Dashboard développé avec Streamlit pour explorer les données et prédictions.

---

### Technologies Utilisées
- **Langage** : Python (3.9 ou supérieur)
- **Bibliothèques Principales** :
  - Web Scraping : `BeautifulSoup`, `Selenium`
  - Data Cleaning : `polars`
  - Machine Learning : `scikit-learn`, `numpy`, `pandas`
  - Visualisation : `Streamlit`, `plotly`
- **Outils** :
  - Gestion de version : `Git`
  - IDE : `VS Code`
  - Plateformes de développement : `GitHub`

---

## Installation

### Prérequis
1. **Python** : Assurez-vous que Python 3.9 ou supérieur est installé.
2. **Git** : Installez Git pour cloner le dépôt.

### Instructions
1. Clonez ce dépôt Git :
   ```bash
   git clone <URL_DU_DEPOT>
   ```
2. Accédez au répertoire cloné :
   ```bash
   cd Pole_Scraping
   ```
3. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Lancer l'Application
1. Exécutez la commande suivante pour lancer l'application Streamlit :
   ```bash
   streamlit run src/app/main.py
   ```
2. Ouvrez votre navigateur et accédez à `http://localhost:8501`.

### Fonctionnalités de l'Application
- **Visualisation des données historiques**.
- **Prédictions en temps réel** pour les qualifications à venir.
- **Tableaux et graphiques interactifs**.

---

## Tests

### Lancer les Tests Unitaires
1. Exécutez la commande suivante pour lancer les tests :
   ```bash
   pytest
   ```
2. Les tests couvrent les modules de scraping, nettoyage, machine learning et visualisation.

---

## Contribution

Les contributions sont les bienvenues ! Pour contribuer :
1. Forkez le dépôt.
2. Créez une branche pour votre fonctionnalité :
   ```bash
   git checkout -b nouvelle-fonctionnalite
   ```
3. Faites vos modifications et committez-les :
   ```bash
   git commit -m "Ajout d'une nouvelle fonctionnalité"
   ```
4. Poussez vos modifications :
   ```bash
   git push origin nouvelle-fonctionnalite
   ```
5. Créez une pull request.

---

## Auteurs

- **Gabin** : Développeur principal.
- **Collaborateurs** : Toute personne ayant contribué au projet.

---

## Licence
Ce projet est sous licence MIT. Veuillez consulter le fichier `LICENSE` pour plus d'informations.

