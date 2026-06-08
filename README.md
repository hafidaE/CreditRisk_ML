# CreditRisk_ML
Application Streamlit de prédiction du niveau de risque de défaut d'un crédit bancaire en se basant sur les données de : 
* Income : Revenu
* Property_value : Valeur de bien
* Loan_amount : montant du pret
* Credit_score 
* dtir1 : Taux d'endettement
* LTV : Loan to amount value
Le modèle utilisé est XGBoost avec les paramètres suivants : {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100, 'scale_pos_weight': 1}
La prédiction retourne la probabilité de défaut d'un crédit si elle est >= 25% => Crédit refusé (Le seuil décisionnel est 0.25 ) 

Le lien de l'application déployée sur Streamlit Cloud : 
https://creditriskml-alcb2luy4fyqt9pxl9vvdv.streamlit.app/

<img width="1378" height="918" alt="image" src="https://github.com/user-attachments/assets/77fab3ea-cc37-43e4-8030-5e5c0f76daaa" />
