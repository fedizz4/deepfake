# frontend/app.py
"""
Interface Streamlit pour la détection de deepfakes
"""
import streamlit as st
import requests
import json
from PIL import Image
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import os
from pathlib import Path

# Configuration de la page
st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .real-box {
        background-color: #D1FAE5;
        border: 2px solid #10B981;
    }
    .fake-box {
        background-color: #FEE2E2;
        border: 2px solid #EF4444;
    }
    .metric-box {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<h1 class="main-header">🕵️ Deepfake Detection System</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #6B7280; margin-bottom: 2rem;'>
    Système de détection de deepfakes utilisant l'IA. Téléchargez une image pour analyser si elle est réelle ou manipulée.
</div>
""", unsafe_allow_html=True)

# Configuration API
API_URL = "http://localhost:8000"  # À changer en production

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    st.title("Navigation")

    page = st.radio(
        "Aller à:",
        ["🏠 Accueil", "📤 Analyser", "📊 Dashboard", "ℹ️ À propos"]
    )

    st.markdown("---")

    # Informations système
    st.subheader("Système")

    # Vérifier la connexion API
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            st.success("✅ API connectée")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Modèle", "✅" if health_data["model_loaded"] else "❌")
            with col2:
                st.metric("Device", health_data["device"])
        else:
            st.error("❌ API non disponible")
    except:
        st.error("❌ Impossible de se connecter à l'API")

    st.markdown("---")

    # Télécharger le rapport
    if st.button("📥 Exporter le rapport"):
        st.info("Fonctionnalité en développement")

# Page d'accueil
if page == "🏠 Accueil":
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<h2 class="sub-header">🎯 À propos du projet</h2>', unsafe_allow_html=True)
        st.markdown("""
        Ce système utilise des modèles de Deep Learning avancés pour détecter les images deepfake.

        **Fonctionnalités principales :**

        - 🔍 **Détection en temps réel** d'images deepfake
        - 📊 **Mesures de confiance** détaillées
        - 🎯 **Précision élevée** grâce à EfficientNet
        - 📈 **Dashboard** de monitoring
        - 🔄 **API REST** pour intégration

        **Technologies utilisées :**

        - **Backend** : PyTorch, FastAPI
        - **Frontend** : Streamlit
        - **Modèles** : EfficientNet, ResNet, Vision Transformer
        - **MLOps** : MLflow, Docker

        **Méthodologie :**

        1. **Extraction** des visages des vidéos
        2. **Préprocessing** et augmentation des données
        3. **Entraînement** de modèles CNN
        4. **Évaluation** avec métriques robustes
        5. **Déploiement** via API et interface web
        """)

    with col2:
        st.markdown('<h2 class="sub-header">📈 Statistiques</h2>', unsafe_allow_html=True)

        # Métriques simulées
        metrics_col1, metrics_col2 = st.columns(2)

        with metrics_col1:
            st.metric("Précision", "96.2%", "1.3%")
            st.metric("Rappel", "94.8%", "0.9%")

        with metrics_col2:
            st.metric("F1-Score", "95.5%", "1.1%")
            st.metric("AUC", "0.983", "0.012")

        st.markdown("---")

        # Graphique de performance
        fig = go.Figure(data=[
            go.Bar(name='Real', x=['Précision', 'Rappel', 'F1'], y=[0.95, 0.97, 0.96]),
            go.Bar(name='Fake', x=['Précision', 'Rappel', 'F1'], y=[0.97, 0.93, 0.95])
        ])

        fig.update_layout(
            title="Performance par classe",
            barmode='group',
            height=300
        )

        st.plotly_chart(fig, use_container_width=True)

# Page d'analyse
elif page == "📤 Analyser":
    st.markdown('<h2 class="sub-header">🔍 Analyse d\'images</h2>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📤 Upload simple", "📁 Upload multiple", "📹 Webcam"])

    with tab1:
        st.markdown("**Téléchargez une seule image pour analyse**")

        uploaded_file = st.file_uploader(
            "Choisissez une image",
            type=['jpg', 'jpeg', 'png'],
            key="single_upload"
        )

        if uploaded_file is not None:
            col1, col2 = st.columns(2)

            with col1:
                # Afficher l'image
                image = Image.open(uploaded_file)
                st.image(image, caption="Image téléchargée", use_column_width=True)

                # Informations image
                st.info(f"**Dimensions :** {image.size[0]}x{image.size[1]} pixels")

            with col2:
                # Bouton d'analyse
                if st.button("🔬 Analyser l'image", type="primary", use_container_width=True):
                    with st.spinner("Analyse en cours..."):
                        try:
                            # Envoyer à l'API
                            files = {"image": uploaded_file.getvalue()}
                            response = requests.post(
                                f"{API_URL}/predict",
                                files={"image": (uploaded_file.name, uploaded_file.getvalue())}
                            )

                            if response.status_code == 200:
                                result = response.json()

                                # Afficher le résultat
                                confidence = result["confidence"]
                                is_fake = result["is_deepfake"]

                                # Barre de confiance
                                st.progress(confidence)
                                st.metric("Confiance", f"{confidence:.2%}")

                                # Boîte de prédiction
                                if is_fake:
                                    st.markdown(
                                        f'<div class="prediction-box fake-box">'
                                        f'<h3 style="color: #EF4444;">⚠️ DEEPFAKE DÉTECTÉ</h3>'
                                        f'<p>Confiance: <b>{confidence:.2%}</b></p>'
                                        f'<p>Temps de traitement: {result["processing_time"]:.3f}s</p>'
                                        f'</div>',
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.markdown(
                                        f'<div class="prediction-box real-box">'
                                        f'<h3 style="color: #10B981;">✅ IMAGE RÉELLE</h3>'
                                        f'<p>Confiance: <b>{confidence:.2%}</b></p>'
                                        f'<p>Temps de traitement: {result["processing_time"]:.3f}s</p>'
                                        f'</div>',
                                        unsafe_allow_html=True
                                    )

                                # Graphique radar
                                fig = go.Figure(data=go.Scatterpolar(
                                    r=[confidence, 1 - confidence, 0.8, 0.6],
                                    theta=['Confiance Fake', 'Confiance Real', 'Fiabilité', 'Précision'],
                                    fill='toself'
                                ))

                                fig.update_layout(
                                    polar=dict(
                                        radialaxis=dict(
                                            visible=True,
                                            range=[0, 1]
                                        )),
                                    showlegend=False,
                                    title="Analyse détaillée",
                                    height=300
                                )

                                st.plotly_chart(fig, use_container_width=True)

                            else:
                                st.error(f"Erreur API: {response.status_code}")

                        except Exception as e:
                            st.error(f"Erreur: {str(e)}")

    with tab2:
        st.markdown("**Téléchargez plusieurs images pour analyse batch**")

        uploaded_files = st.file_uploader(
            "Choisissez plusieurs images",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            key="batch_upload"
        )

        if uploaded_files and len(uploaded_files) > 0:
            if st.button("🔬 Analyser le batch", type="primary"):
                with st.spinner(f"Analyse de {len(uploaded_files)} images..."):
                    try:
                        # Préparer les fichiers
                        files = []
                        for uploaded_file in uploaded_files:
                            files.append(("images", (uploaded_file.name, uploaded_file.getvalue())))

                        # Envoyer à l'API
                        response = requests.post(
                            f"{API_URL}/predict/batch",
                            files=files
                        )

                        if response.status_code == 200:
                            results = response.json()

                            # Créer un DataFrame
                            df_data = []
                            for pred in results["predictions"]:
                                if "error" not in pred:
                                    df_data.append({
                                        "Fichier": pred["filename"],
                                        "Prédiction": pred["prediction"],
                                        "Confiance": f"{pred['confidence']:.2%}",
                                        "Deepfake": "Oui" if pred["is_deepfake"] else "Non"
                                    })

                            if df_data:
                                df = pd.DataFrame(df_data)

                                # Afficher le tableau
                                st.dataframe(df, use_container_width=True)

                                # Statistiques
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    real_count = len(df[df["Deepfake"] == "Non"])
                                    st.metric("Images réelles", real_count)
                                with col2:
                                    fake_count = len(df[df["Deepfake"] == "Oui"])
                                    st.metric("Deepfakes", fake_count)
                                with col3:
                                    st.metric("Temps total", f"{results['processing_time']:.2f}s")

                                # Graphique
                                fig = go.Figure(data=[go.Pie(
                                    labels=['Réelles', 'Deepfakes'],
                                    values=[real_count, fake_count],
                                    hole=.3
                                )])

                                fig.update_layout(
                                    title="Répartition des prédictions",
                                    height=300
                                )

                                st.plotly_chart(fig, use_container_width=True)

                                # Bouton d'export
                                csv = df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="📥 Télécharger les résultats (CSV)",
                                    data=csv,
                                    file_name=f"deepfake_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )

                        else:
                            st.error(f"Erreur API: {response.status_code}")

                    except Exception as e:
                        st.error(f"Erreur: {str(e)}")

    with tab3:
        st.markdown("**Utilisez votre webcam pour capture et analyse**")

        # Note: Streamlit Cloud ne supporte pas l'accès à la webcam
        st.info("""
        ⚠️ **Cette fonctionnalité nécessite un accès à la webcam.**

        Sur Streamlit Cloud, cette fonctionnalité n'est pas disponible.
        Pour l'utiliser localement, exécutez:

        ```bash
        streamlit run app.py
        ```

        Assurez-vous d'avoir installé `opencv-python`:
        ```bash
        pip install opencv-python
        ```
        """)

        # Code pour la webcam (désactivé sur cloud)
        # picture = st.camera_input("Prendre une photo")
        # if picture:
        #     # Même traitement que l'upload simple
        #     pass

# Page Dashboard
elif page == "📊 Dashboard":
    st.markdown('<h2 class="sub-header">📊 Tableau de bord</h2>', unsafe_allow_html=True)

    # Métriques globales
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total images", "1,250", "12%")
    with col2:
        st.metric("Précision globale", "96.2%", "1.3%")
    with col3:
        st.metric("Faux positifs", "3.8%", "-0.5%")
    with col4:
        st.metric("Temps réponse", "0.15s", "0.02s")

    st.markdown("---")

    # Graphiques
    col1, col2 = st.columns(2)

    with col1:
        # Courbe ROC
        st.subheader("Courbe ROC")

        # Données simulées
        fpr = np.linspace(0, 1, 100)
        tpr = np.sqrt(fpr)  # ROC simulée

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve', line=dict(width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random',
                                 line=dict(dash='dash', color='gray')))

        fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=400,
            annotations=[dict(
                x=0.7, y=0.3,
                text=f"AUC = 0.983",
                showarrow=False,
                font=dict(size=14)
            )]
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Matrice de confusion
        st.subheader("Matrice de confusion")

        # Données simulées
        cm = [[235, 12], [8, 245]]

        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Prédit Real', 'Prédit Fake'],
            y=['Vrai Real', 'Vrai Fake'],
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 16},
            colorscale='Blues'
        ))

        fig.update_layout(
            height=400,
            xaxis_title="Prédiction",
            yaxis_title="Vérité"
        )

        st.plotly_chart(fig, use_container_width=True)

    # Comparaison des modèles
    st.markdown("---")
    st.subheader("🔍 Comparaison des modèles")

    models_data = {
        'Modèle': ['EfficientNet-B0', 'ResNet-50', 'Xception', 'Vision Transformer'],
        'Accuracy': [0.962, 0.951, 0.958, 0.945],
        'AUC': [0.983, 0.975, 0.980, 0.972],
        'Temps (ms)': [15, 22, 18, 35],
        'Paramètres (M)': [5.3, 25.6, 22.9, 22.1]
    }

    df_models = pd.DataFrame(models_data)
    st.dataframe(df_models, use_container_width=True)

    # Graphique de comparaison
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Accuracy', 'AUC', 'Temps d\'inférence', 'Nombre de paramètres')
    )

    # Accuracy
    fig.add_trace(
        go.Bar(x=df_models['Modèle'], y=df_models['Accuracy'], name='Accuracy'),
        row=1, col=1
    )

    # AUC
    fig.add_trace(
        go.Bar(x=df_models['Modèle'], y=df_models['AUC'], name='AUC'),
        row=1, col=2
    )

    # Temps
    fig.add_trace(
        go.Bar(x=df_models['Modèle'], y=df_models['Temps (ms)'], name='Temps'),
        row=2, col=1
    )

    # Paramètres
    fig.add_trace(
        go.Bar(x=df_models['Modèle'], y=df_models['Paramètres (M)'], name='Paramètres'),
        row=2, col=2
    )

    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# Page À propos
elif page == "ℹ️ À propos":
    st.markdown('<h2 class="sub-header">ℹ️ À propos du projet</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("""
        ## 📚 Contexte académique

        Ce projet a été développé dans le cadre du module **Deep Learning Python** de Terminale Data Science.

        **Objectifs pédagogiques :**

        1. **Maîtriser les architectures de Deep Learning modernes**
        2. **Développer une méthodologie expérimentale rigoureuse**
        3. **Acquérir des compétences en MLOps et déploiement**
        4. **Comprendre les enjeux éthiques de l'IA**

        ## 🎯 Spécifications techniques

        **Livrables obligatoires :**

        - ✅ **Repository Git** avec structure claire
        - ✅ **Modèle entraîné** sauvegardé
        - ✅ **API de service** (FastAPI)
        - ✅ **Interface utilisateur** (Streamlit)
        - ✅ **Traçabilité MLflow** complète
        - ✅ **Conteneurisation Docker**
        - ✅ **Tests unitaires** et d'intégration

        **Architecture technique :**

        - **Backend** : PyTorch, FastAPI, MLflow
        - **Frontend** : Streamlit, Plotly
        - **Déploiement** : Docker, docker-compose
        - **Monitoring** : MLflow tracking, logging

        ## 👥 Équipe

        - **Étudiant(e)** : [Votre nom]
        - **Encadrant** : Haythem Ghazouani
        - **Niveau** : Terminale Data Science
        - **Institution** : [Votre école]

        ## 📅 Planning

        - **Semaine 1** : Analyse des données et EDA
        - **Semaine 2** : Preprocessing et baseline
        - **Semaine 3** : Entraînement des modèles
        - **Semaine 4** : Expérimentations comparatives
        - **Semaine 5** : API et interface utilisateur
        - **Semaine 6** : Tests et documentation

        ## ⚠️ Limitations et éthique

        **Limitations techniques :**

        - Modèle entraîné sur FaceForensics++ (limité à 4 méthodes de manipulation)
        - Performances peuvent varier sur des deepfakes récents
        - Nécessite des visages bien détectés pour une analyse précise

        **Considérations éthiques :**

        - Utilisation responsable de la technologie
        - Respect de la vie privée
        - Transparence sur les limites du système
        - Documentation des biais potentiels

        ## 📞 Contact

        Pour toute question concernant ce projet :

        - **Email** : [votre.email@example.com]
        - **GitHub** : [lien vers votre repository]
        - **LinkedIn** : [votre profil LinkedIn]
        """)

    with col2:
        # QR Code pour le repository (simulé)
        st.image("https://api.qrserver.com/v1/create-qr-code/?size=150x150&data=https://github.com/votre-repo",
                 caption="Repository GitHub")

        st.markdown("---")

        # Badges
        st.markdown("**Technologies utilisées :**")

        badges = {
            "PyTorch": "https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white",
            "FastAPI": "https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi",
            "Streamlit": "https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white",
            "Docker": "https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white",
            "MLflow": "https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white"
        }

        for tech, badge_url in badges.items():
            st.markdown(f"![{tech}]({badge_url})")

        st.markdown("---")

        # Version
        st.info("""
        **Version :** 1.0.0  
        **Dernière mise à jour :**  
        """ + datetime.now().strftime("%d/%m/%Y"))

        # Bouton pour les logs
        if st.button("📋 Voir les logs système"):
            try:
                response = requests.get(f"{API_URL}/logs/latest")
                if response.status_code == 200:
                    st.text_area("Logs système", response.text, height=200)
                else:
                    st.error("Impossible de récupérer les logs")
            except:
                st.error("API non disponible")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <div style='text-align: center; color: #6B7280; font-size: 0.9rem;'>
    Projet Deep Learning Python - Terminale Data Science<br>
    © 2024 - Développé avec ❤️ et 🤖
    </div>
    """, unsafe_allow_html=True)