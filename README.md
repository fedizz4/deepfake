# 🧠 Deepfake Facial Detection System

📅 **Date** : 30 janvier 2026  

## 👥 Groupe
- **Fedi Zran**
- **Fadwa Zran**
- **Asma Bargaoui**

---

## 📌 Contexte du projet

Avec l’évolution rapide des techniques d’intelligence artificielle générative, les **deepfakes** sont devenus de plus en plus réalistes et difficiles à détecter. Ces manipulations visuelles posent des problèmes majeurs en matière de **désinformation**, **sécurité**, **vie privée** et **éthique**.

Ce projet s’inscrit dans le cadre du **module Deep Learning** et vise à appliquer des techniques modernes de **Computer Vision** afin de détecter automatiquement les visages manipulés (deepfakes) à partir d’images.

---

## 🎯 Objectif du projet

L’objectif principal est de **concevoir, entraîner et déployer un système intelligent capable de détecter les deepfakes faciaux**, tout en respectant une démarche scientifique rigoureuse et des bonnes pratiques MLOps.

Les objectifs spécifiques sont :

- Développer un **modèle de Deep Learning performant** pour la détection de deepfakes
- Fournir une **prédiction claire** :  
  → *Image réelle* ou *Image falsifiée (Deepfake)* avec un **score de confiance**
- Rendre le modèle **interprétable** grâce à des techniques d’explicabilité (Grad-CAM)
- Déployer le modèle via une **API FastAPI**
- Proposer une **interface utilisateur interactive**
- Assurer la **traçabilité et la reproductibilité** des expériences (MLflow + Docker)

---

## 🧩 Approche et méthodologie

Le projet est structuré selon les étapes suivantes :

### 1️⃣ Analyse et préparation des données
- Utilisation de datasets spécialisés en deepfake (FaceForensics++, Celeb-DF)
- Extraction de frames depuis des vidéos
- Détection et recadrage des visages
- Normalisation et augmentation des données

---

### 2️⃣ Modélisation
- Utilisation de modèles de vision modernes :
  - **EfficientNet** (modèle principal)
  - Comparaison possible avec **Vision Transformer (ViT)**
- Entraînement supervisé pour une classification binaire :
  - `Real` / `Deepfake`

---

### 3️⃣ Évaluation
- Mesures de performance :
  - Accuracy
  - AUC-ROC
  - Precision / Recall
  - Confusion Matrix
- Analyse des erreurs et tests de robustesse

---

### 4️⃣ Interprétabilité
- Génération de **cartes de chaleur Grad-CAM**
- Visualisation des zones de l’image utilisées par le modèle pour prendre sa décision

---

### 5️⃣ Déploiement et MLOps
- Backend : **FastAPI**
- Interface utilisateur : **Streamlit**
- Suivi des expériences : **MLflow**
- Conteneurisation : **Docker & Docker Compose**
- Architecture modulaire et reproductible

---

## 🏗️ Architecture du projet

# Deepfake Detection Project

EfficientNet + FastAPI + Streamlit + MLflow

deepfake-detector/
│
├── data/ # Données brutes et traitées
├── src/ # Entraînement et évaluation du modèle
├── api/ # API FastAPI
├── frontend/ # Interface Streamlit
├── docker/ # Docker & docker-compose
├── mlruns/ # Logs MLflow
├── tests/ # Tests unitaires
├── requirements.txt
└── README.md


---

## ✅ Résultat attendu

À la fin du projet, le système permettra :

- D’uploader une image contenant un visage
- D’obtenir une prédiction fiable (*réelle ou deepfake*)
- De visualiser un score de confiance
- D’interpréter la décision du modèle
- De démontrer un pipeline complet de Deep Learning **de la donnée au déploiement**

---

## ⚖️ Considérations éthiques

Une attention particulière est portée sur :
- L’usage responsable des datasets
- Les biais potentiels du modèle
- La transparence des résultats
- Les limites du système développé

---

## 📌 Conclusion

Ce projet vise à rapprocher les étudiants des **problématiques réelles de l’IA en production**, en combinant rigueur scientifique, compétences techniques et réflexion éthique.

---
