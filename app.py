import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. La fonction obligatoire pour le pipeline (à laisser tout en haut)
def imputer_ltv(X):
    X_copy = X.copy()
    ltv_theorique = (X_copy['loan_amount'] / X_copy['property_value']) * 100
    X_copy['LTV'] = X_copy['LTV'].fillna(ltv_theorique)
    return X_copy

# 2. Chargement très simple du modèle
pipeline = joblib.load("pipeline_credit_xgboost.joblib")

# 3. Titre de l'application
st.title("Application Score de Crédit")

# 4. Formulaire de saisie des données (les unes en dessous des autres)
st.subheader("Saisie des données du client")

income = st.number_input("Revenu", value=3500)
property_value = st.number_input("Valeur du bien", value=200000)
dtir1 = st.number_input("Taux endettement (dtir1)", value=33.0)
ltv = st.number_input("LTV (laisser 0 pour calcul automatique)", value=0.0)
loan_amount = st.number_input("Montant du pret", value=150000)
credit_score = st.number_input("Credit Score", value=650)

# Gestion du LTV automatique si l'étudiant laisse 0
if ltv == 0.0:
    ltv = np.nan

# 5. Bouton pour lancer les calculs
if st.button("Calculer la prédiction"):
    
    # Création du dictionnaire de données
    donnees = {
        'income': income,
        'property_value': property_value,
        'dtir1': dtir1,
        'LTV': ltv,
        'loan_amount': loan_amount,
        'Credit_Score': credit_score
    }
    
    # Transformation en DataFrame (1 seule ligne)
    df_saisie = pd.DataFrame([donnees])
    
    # --- CONSIGNE 1 : Affichage des données saisies ---
    st.write("Données reçues pour l'analyse :")
    st.dataframe(df_saisie)
    
    # --- CONSIGNE 2 : Calcul de la prédiction et probabilité ---
    # predict_proba renvoie une matrice, on prend la ligne 0, colonne 1
    proba_defaut = pipeline.predict_proba(df_saisie)[0, 1]
    
    # Conversion en type float standard pour éviter les bugs d'affichage
    proba_defaut = float(proba_defaut)
    
    # Affichage de la probabilité (Niveau de confiance)
    st.write("Probabilité de défaut (Niveau de risque) :", proba_defaut)
    
    # Affichage de la décision finale (Prédiction)
    if proba_defaut > 0.25:
        st.write("Résultat : Crédit Refusé")
    else:
        st.write("Résultat : Crédit Accordé")