from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# 1. Fonction obligatoire exigée par ton pipeline
def imputer_ltv(X):
    X_copy = X.copy()
    ltv_theorique = (X_copy['loan_amount'] / X_copy['property_value']) * 100
    X_copy['LTV'] = X_copy['LTV'].fillna(ltv_theorique)
    return X_copy

# 2. Chargement du modèle au démarrage
pipeline = joblib.load("pipeline_credit_xgboost.joblib")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction_texte = None
    probabilite_texte = None
    donnees_saisies = None

    if request.method == "POST":
        # Récupération des données du formulaire HTML
        income = float(request.form.get("income"))
        property_value = float(request.form.get("property_value"))
        dtir1 = float(request.form.get("dtir1"))
        loan_amount = float(request.form.get("loan_amount"))
        credit_score = float(request.form.get("credit_score"))
        
        # Gestion du LTV automatique s'il n'est pas rempli
        ltv_input = request.form.get("ltv")
        ltv = float(ltv_input) if ltv_input else np.nan

        # Dictionnaire pour l'affichage des données saisies (Consigne)
        donnees_saisies = {
            "Revenu": income, "Valeur du bien": property_value, 
            "DTIR": dtir1, "LTV": ltv if not np.isnan(ltv) else "Calculé par l'IA", 
            "Montant": loan_amount, "Score": credit_score
        }

        # Création du DataFrame dans l'ordre du pipeline
        df_input = pd.DataFrame([{
            'income': income, 'property_value': property_value, 'dtir1': dtir1,
            'LTV': ltv, 'loan_amount': loan_amount, 'Credit_Score': credit_score
        }])

        # Prédiction
        prob = float(pipeline.predict_proba(df_input)[0, 1])
        probabilite_texte = f"{prob * 100:.2f} %"

        if prob >= 0.25:
            prediction_texte = "CRÉDIT REFUSÉ (Risque trop élevé)"
        else:
            prediction_texte = "CRÉDIT ACCORDÉ"

    return render_template("index.html", 
                           prediction=prediction_texte, 
                           probabilite=probabilite_texte, 
                           donnees=donnees_saisies)

# Lancement local (Utile pour tester sur ton PC)
if __name__ == "__main__":
    app.run(debug=True)