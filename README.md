# Deep-Learning-project
Ce projet implémente une architecture Capsule Network (CapsNet) en utilisant PyTorch pour la classification des chiffres manuscrits du dataset MNIST. Il inclut des techniques d'augmentation de données, un algorithme de routage dynamique et une reconstruction des images pour améliorer la robustesse du modèle.

# Fonctionnalités
CapsNet avec routing dynamique pour une meilleure représentation des features.  
Reconstruction des images à partir des vecteurs de capsules pour une régularisation supplémentaire.  
Optimisation avec Adam et choix optimal des hyperparamètres.  
Augmentation des données (translations ±2 pixels) pour améliorer la généralisation.  

# Structure du projet
Projet  
│── data/                  # Dossier contenant le dataset MNIST (impossile de l'importer sur git trop lourd)  
│── run/                   # Implémentation du modèle CapsNet  
│── load_data.py           # Script principal d'entraînement  
│── test_1epoch.py         # recherche hyperparamètres (grid search) sur 1 epoch  
│── test_overfit.py        # test pour voir si le modèle overfit de manière intentionnelle  
│── README.md              # Documentation du projet  

# Installation et exécution
1) Cloner le projet  
git clone https://github.com/votre-utilisateur/CapsNet-MNIST.git  
cd CapsNet-MNIST  

2) Installer les dépendances  
pip install -r requirements.txt  

3) Lancer l'entraînement  
python load_data.py  
 
4) Courbes d'apprentissage (Tensorboard)  
dans un terminal: $tensorboard --logdir=runs  

# Hyperparamètres optimaux  

Learning Rate: 0.0003  
Reconstruction Loss Weight: 0.0007  
Batch Size:128  
Epochs: 100 (avec early stopping)  

# Résultats  
CapsNet (Article): accuracy = 99.75%  
Mon implémentation: accuracy = 99.47%  
