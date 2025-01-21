from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import pandas as pd
import numpy as np

from src.ml.lib_fonctions_ML import (
    modif_data,
    split_train_test,
    preprocess_data, 
    boosting_entrainement,
    boosting_test
)

def test_modif_data():
    data = pd.DataFrame({
        'Temps_sec': [120.0, 150.0, 90.0],
        'Longueur_km': [3.0, 5.0, 2.0]
    })

    result = modif_data(data)

    assert 'Allure' in result.columns, "La colonne 'Allure' doit être ajoutée."
    assert result['Allure'].iloc[0] == 40.0, "Le calcul de l'allure est incorrect."
    assert len(result) == 3, "Le nombre de lignes doit rester identique."
    assert result['Allure'].iloc[1] == 30.0, "Le calcul de l'allure est incorrect pour la deuxième ligne."

def test_plantage_modif_data():
    import pandas as pd

    colonnes_manquantes = pd.DataFrame({
        'Temps_sec': [120.0, 150.0, 90.0],
        'Vitesse': [3.0, 5.0, 2.0]
    })
    try:
        modif_data(colonnes_manquantes)
    except ValueError as e:
        assert "Les colonnes 'Temps_sec' et 'Longueur_km'" in str(e), "Erreur non détectée pour colonnes manquantes."

    vide = pd.DataFrame()
    try:
        modif_data(vide)
    except ValueError as e:
        assert "Les colonnes 'Temps_sec' et 'Longueur_km'" in str(e), "Erreur non détectée pour un DataFrame vide."

    invalid_data = pd.DataFrame({
        'Temps_sec': [120.0, None, 90.0],
        'Longueur_km': [3.0, 5.0, None]
    })
    try:
        modif_data(invalid_data)
    except Exception as e:
        assert "division" in str(e).lower() or "None" in str(e), "Erreur non détectée pour des valeurs nulles."

def test_split_train_test():
    data = pd.DataFrame({
        'Ligne1': [1, 2, 3, 4, 5],
        'Ligne 2': [5, 4, 3, 2, 1],
        'Target': [0, 1, 0, 1, 0]
    })

    X_train, X_test, y_train, y_test = split_train_test(data, variable='Target')

    assert len(X_train) == 4, "Le DataFrame d'entraînement doit contenir 80% des données."
    assert len(X_test) == 1, "Le DataFrame de test doit contenir 20% des données."
    assert len(y_train) == 4, "La série cible d'entraînement doit contenir 80% des données."
    assert len(y_test) == 1, "La série cible de test doit contenir 20% des données."
    assert 'Target' not in X_train.columns, "La colonne cible ne doit pas être présente dans les caractéristiques."
    assert 'Target' not in X_test.columns, "La colonne cible ne doit pas être présente dans les caractéristiques."

def test_plantage_split_train_test():
    missing_target_data = pd.DataFrame({
        'Ligne1': [1, 2, 3],
        'Ligne2': [4, 5, 6]
    })
    try:
        split_train_test(missing_target_data, variable='Target')
    except KeyError as e:
        assert "['Target'] not found in axis" in str(e), "Erreur non détectée pour une colonne cible manquante."

    small_data = pd.DataFrame({
        'Ligne1': [1],
        'Ligne2': [2],
        'Target': [0]
    })
    try:
        split_train_test(small_data, variable='Target')
    except ValueError as e:
        assert "test_size=" in str(e), "Erreur non détectée pour un jeu de données trop petit."

def test_preprocess_data():
    donnees = pd.DataFrame({
        'Pilote': ['Hamilton', 'Verstappen'],
        'GP': ['Monaco', 'Silverstone'],
        'Chassis': ['Mercedes', 'Red Bull'],
        'Circuit': ['Lent', 'Rapide'],
        'Saison': [2024, 2025],
        'Numeros_gp': [7, 8],
        'Moteur_Ferrari': [0, 0],
        'Moteur_Honda': [0, 1],
        'Moteur_Mercedes': [1, 0],
        'Moteur_Renault': [0, 0],
        'Pluie': [0, 1],
        'Nuit': [0, 0],
        'Circuit_Lent': [1, 0],
        'Circuit_Moderé': [0, 0],
        'Circuit_Rapide': [0, 1],
        'Virages': [19, 15],
        'position_gp_saison_precedente': [2, 1],
        'chrono_gp_saison_precedente': [70.2, 69.8]
    })

    preprocesseur = preprocess_data()

    donnees_transformees = preprocesseur.fit_transform(donnees)

    assert isinstance(preprocesseur, ColumnTransformer), "La fonction doit renvoyer un ColumnTransformer."
    assert donnees_transformees.shape[1] > 0, "Les données transformées doivent contenir des colonnes."
    assert np.any(donnees_transformees), "Les données transformées ne doivent pas être vides."

def test_plantage_preprocess_data():
    donnees_vides = pd.DataFrame()
    preprocesseur = preprocess_data()
    try:
        preprocesseur.fit_transform(donnees_vides)
    except ValueError as e:
        assert "A given column is not a column of the dataframe" in str(e), \
            "Erreur non détectée pour un DataFrame vide."

    donnees_incompletes = pd.DataFrame({
        'Pilote': ['Hamilton'],
        'GP': ['Monaco']
    }) 
    try:
        preprocesseur.fit_transform(donnees_incompletes)
    except ValueError as e:
        assert "A given column is not a column of the dataframe" in str(e), \
            "Erreur non détectée pour des colonnes manquantes."

def test_succes_boosting():
    X_train = pd.DataFrame({
        'Pilote': ['Hamilton', 'Verstappen', 'Leclerc', 'Alonso', 'Sainz'],
        'GP': ['Monaco', 'Monaco', 'Monaco', 'Monaco', 'Monaco'],
        'Saison': [2025, 2025, 2025, 2025, 2025],
        'Longueur_km': [3.337, 3.337, 3.337, 3.337, 3.337],
        'Temps_sec': [70.567, 71.234, 72.890, 73.500, 74.200]
    })
    y_train = pd.Series([1.23, 1.34, 1.45, 1.56, 1.67], name="Allure")

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Pilote', 'GP']),
            ('num', StandardScaler(), ['Saison', 'Longueur_km', 'Temps_sec'])
        ]
    )

    pipeline, params = boosting_entrainement(X_train, y_train, preprocessor)

    assert pipeline is not None, "Le pipeline entraîné est vide."
    assert isinstance(params, pd.DataFrame), "Les meilleurs hyperparamètres doivent être retournés sous forme de DataFrame."
    assert not params.empty, "Le DataFrame des meilleurs hyperparamètres ne doit pas être vide."

def test_plantage_boosting_entrainement():
    donnees_vides = pd.DataFrame()
    y_vide = pd.Series(dtype=float)
    preprocesseur = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Pilote', 'GP']),
            ('num', StandardScaler(), ['Saison', 'Longueur_km', 'Temps_sec'])
        ]
    )
    try:
        boosting_entrainement(donnees_vides, y_vide, preprocesseur)
    except ValueError as e:
        assert "Cannot have number of splits" in str(e), \
            "Erreur non détectée pour des données d'entraînement vides."

def test_boosting_test():
    best_pipeline = joblib.load("data/ml/models/model_boosting_allure.pkl")

    X_test = pd.DataFrame({
        'Pilote': ['Hamilton', 'Verstappen', 'Leclerc'],
        'GP': ['Monaco', 'Monaco', 'Monaco'],
        'Saison': [2025, 2025, 2025],
        'Longueur_km': [3.337, 3.337, 3.337],
        'Temps_sec': [70.567, 71.234, 72.890],
        'Numeros_gp': [7, 7, 7],
        'Chassis': ['Mercedes', 'Red Bull', 'Ferrari'],
        'Moteur_Ferrari': [0, 0, 1],
        'Moteur_Honda': [0, 1, 0],
        'Moteur_Mercedes': [1, 0, 0],
        'Moteur_Renault': [0, 0, 0],
        'Pluie': [0, 0, 0],
        'Nuit': [0, 0, 0],
        'Circuit': ['Monaco', 'Monaco', 'Monaco'],
        'Circuit_Lent': [0, 0, 0],
        'Circuit_Moderé': [1, 1, 1],
        'Circuit_Rapide': [0, 0, 0],
        'Virages': [18, 18, 18],
        'position_gp_saison_precedente': [6, 7, 8],
        'chrono_gp_saison_precedente': [70.567, 71.234, 72.890]
    })
    y_test = pd.Series([1.23, 1.34, 1.45], name="Allure")

    metrics_df, results_df = boosting_test(best_pipeline, X_test, y_test)

    assert "Metric" in metrics_df.columns, "Les métriques ne contiennent pas la colonne 'Metric'."
    assert "Value" in metrics_df.columns, "Les métriques ne contiennent pas la colonne 'Value'."
    assert len(metrics_df) == 3, "Les métriques devraient inclure RMSE, MAE et R²."

    assert "Pilote" in results_df.columns, "Les résultats ne contiennent pas la colonne 'Pilote'."
    assert "Réalité" in results_df.columns, "Les résultats ne contiennent pas la colonne 'Réalité'."
    assert "Prédiction" in results_df.columns, "Les résultats ne contiennent pas la colonne 'Prédiction'."

def test_plantage_boosting_test():
    best_pipeline = joblib.load("data/ml/models/model_boosting_allure.pkl")

    X_test_incomplet = pd.DataFrame({
        'Pilote': ['Hamilton', 'Verstappen', 'Leclerc'],
        'GP': ['Monaco', 'Monaco', 'Monaco'],
        'Saison': [2025, 2025, 2025],
        'Longueur_km': [3.337, 3.337, 3.337],
        'Temps_sec': [70.567, 71.234, 72.890]
    })
    y_test = pd.Series([1.23, 1.34, 1.45], name="Allure")
    try:
        boosting_test(best_pipeline, X_test_incomplet, y_test)
    except ValueError as e:
        assert "columns are missing" in str(e), \
            "Erreur non détectée pour des colonnes manquantes dans les données de test."