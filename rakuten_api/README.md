📦 API Rakuten France
Classification Multimodale de Produits E-commerce

1. Introduction et Objectifs
Ce projet a été réalisé dans le cadre de la formation ML Engineer chez DataScientest. L’objectif est de déployer une solution de Deep Learning capable de classifier les produits du catalogue Rakuten France en utilisant des données textuelles et des images.
Étapes clés :
Déploiement des modèles de classification via une API FastAPI.
Gestion et authentification des utilisateurs via MongoDB
Conteneurisation complète avec Docker et Docker Compose.
Automatisation des tests d'authentification, d'autorisation et de prédiction.

2. Architecture de l'API (FastAPI)
L'API utilise le protocole OAuth2 avec des JWT tokens pour la sécurité. La durée de validité des tokens est de 30 minutes.
2.1 Points d'accès principaux (Endpoints)
Catégorie
Endpoint
Description
Général
/Root
Vérification de l'état de l'API.


/token
Obtention du jeton d'accès (Email/Password).
Admin
/Admin/add_user
Création d'utilisateurs (Rôles: admin, dev, user).


/Admin/list_users
Liste des utilisateurs enregistrés.
User
/Users/register_user
Auto-enregistrement des nouveaux utilisateurs.


/Users/Me
Informations sur l'utilisateur connecté.


3. Modèles de Prédiction
L'API permet d'effectuer des prédictions selon trois modes:
3.1 Classification par Texte
  Modèles : Conv1D et Simple DNN.
  Entrées : Désignation (obligatoire) et Description (optionnelle).
3.2 Classification par Image
  Modèles : Xception et InceptionV3.
  Entrées : Fichier image du produit.

3.3 Classification Multimodale (Texte + Image)
  Combinations : Fusion de Conv1D, Simple DNN et Xception/Inception.
  Sortie : Renvoie la prédiction finale combinée ainsi que les résultats individuels de chaque modèle.

5. Conteneurisation et Déploiement
Le projet est entièrement orchestré via Docker Compose, qui gère les services suivants:

fastApi_rakuten : Le cœur de l'application.
rakuten-mongo : Base de données pour les utilisateurs.
Services de Tests : 3 conteneurs dédiés (Authentication, Authorization, Prediction) qui génèrent un fichier api_tests.log.

7. Guide des Tests et Utilisateurs
Utilisateurs de Test pré-enregistrés:

Admin : admin_account1@example.com | MDP: adminsecret1.
User : johndoe@example.com | MDP: secret2.

Validation des prédictions :
Un test est considéré comme "SUCCESS" si la classe prédite (classe, label et précision) correspond à la catégorie attendue du catalogue Rakuten.

9. Structure du Répertoire GitHub
/app : Code source Python de l'API.
/tests_images : Dossiers contenant les Dockerfiles et scripts pour les tests d'authentification, d'autorisation et de prédiction.
docker-compose.yml : Fichier de configuration des services.

The API est disponible à l'adresse :  
[http://localhost:8000](http://localhost:8000)

Documentation et interface :  
[http://localhost:8000/docs](http://localhost:8000/docs)
