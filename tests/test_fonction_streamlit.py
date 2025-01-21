import pytest
import polars as pl
import pandas as pd
import joblib
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression

from src.app.fonction_streamlit_app import (
    historic_positions,
    position_moyenne,
    historic_positions_par_gp,
    position_moyenne_par_GP,
    position_moyenne_par_saison,
    stats_pilotes_saisons,
    stats_ecuries_saisons,
    correlation_matrix,
    importance_variable,
    preparation_data_prediction,
    prediction_grille,
    ordonner_prediction
)

def test_historic_positions():
    sample_data = {
        "Pilote": ["Hamilton", "Verstappen", "Hamilton", "Verstappen", "Hamilton"],
        "Position": [1, 2, 3, 4, 5],
        "GP": ["Australia", "Bahrain", "China", "Monaco", "Canada"],
        "Saison": [2023, 2023, 2023, 2023, 2023],
    }
    sample_df = pl.DataFrame(sample_data)

    fig = historic_positions("Hamilton", 3, sample_df)
    assert isinstance(fig, go.Figure), "La fonction doit retourner un objet de type plotly.graph_objects.Figure."

    fig = historic_positions("Hamilton", 2, sample_df)
    assert len(fig.data[0].x) == 2, "Le graphique doit inclure 2 points correspondant aux 2 derniers GP."
    assert len(fig.data[0].y) == 2, "Le graphique doit inclure les positions des 2 derniers GP."

def test_platange_historic_positions():
    sample_data = {
        "Pilote": ["Hamilton", "Verstappen", "Hamilton", "Verstappen", "Hamilton"],
        "Position": [1, 2, 3, 4, 5],
        "GP": ["Australia", "Bahrain", "China", "Monaco", "Canada"],
        "Saison": [2023, 2023, 2023, 2023, 2023],
    }
    sample_df = pl.DataFrame(sample_data)

    with pytest.raises(ValueError, match="Le pilote 'Leclerc' n'est pas présent.*"):
        historic_positions("Leclerc", 3, sample_df)

    with pytest.raises(ValueError, match="Le nombre de Grands Prix spécifié.*"):
        historic_positions("Hamilton", 10, sample_df)

def test_position_moyenne_success():
    sample_data = {
        "Pilote": ["Hamilton", "Verstappen", "Hamilton", "Verstappen", "Hamilton"],
        "Position": [1, 2, 3, 4, 5],
        "GP": ["Australia", "Bahrain", "China", "Monaco", "Canada"],
        "Saison": [2023, 2023, 2023, 2023, 2023],
    }
    sample_df = pl.DataFrame(sample_data)

    moyenne = position_moyenne("Hamilton", 3, sample_df)
    assert moyenne == 3.0, "La moyenne des positions pour Hamilton sur les 3 derniers GP doit être 3.0."

    moyenne = position_moyenne("Hamilton", 2, sample_df)
    assert moyenne == 4.0, "La moyenne des positions pour Hamilton sur les 2 derniers GP doit être 4.0."

def test_plantage_position_moyenne():
    sample_data = {
        "Pilote": ["Hamilton", "Verstappen", "Hamilton", "Verstappen", "Hamilton"],
        "Position": [1, 2, 3, 4, 5],
        "GP": ["Australia", "Bahrain", "China", "Monaco", "Canada"],
        "Saison": [2023, 2023, 2023, 2023, 2023],
    }
    sample_df = pl.DataFrame(sample_data)

    with pytest.raises(ValueError, match="Le pilote 'Leclerc' n'est pas présent.*"):
        position_moyenne("Leclerc", 3, sample_df)

    with pytest.raises(ValueError, match="Le nombre de Grands Prix spécifié.*"):
        position_moyenne("Hamilton", 10, sample_df)

    empty_df = pl.DataFrame({"Pilote": [], "Position": [], "GP": [], "Saison": []})
    with pytest.raises(ValueError, match="Le pilote 'Hamilton' n'est pas présent.*"):
        position_moyenne("Hamilton", 3, empty_df)

def test_historic_positions_par_gp():
    sample_data = {
        "Pilote": ["Hamilton", "Verstappen", "Hamilton", "Verstappen", "Hamilton"],
        "Position": [1, 2, 3, 4, 5],
        "GP": ["Australia", "Bahrain", "China", "Monaco", "Canada"],
        "Saison": [2023, 2023, 2023, 2023, 2023],
    }
    sample_df = pl.DataFrame(sample_data)

    fig = historic_positions_par_gp("Hamilton", "Canada", sample_df)
    assert isinstance(fig, go.Figure), "La fonction doit retourner un objet de type plotly.graph_objects.Figure."
    assert len(fig.data[0].x) == 1, "Le graphique doit inclure un seul point correspondant au GP spécifié."
    assert len(fig.data[0].y) == 1, "Le graphique doit inclure la position du GP spécifié."

def test_plantage_historic_positions_par_gp_failure():
    sample_data = {
        "Pilote": ["Hamilton", "Verstappen", "Hamilton", "Verstappen", "Hamilton"],
        "Position": [1, 2, 3, 4, 5],
        "GP": ["Australia", "Bahrain", "China", "Monaco", "Canada"],
        "Saison": [2023, 2023, 2023, 2023, 2023],
    }
    sample_df = pl.DataFrame(sample_data)

    with pytest.raises(ValueError, match="Le pilote 'Leclerc' n'est pas présent.*"):
        historic_positions_par_gp("Leclerc", "Canada", sample_df)

    with pytest.raises(ValueError, match="Le Grand Prix 'Monza' n'est pas disponible.*"):
        historic_positions_par_gp("Hamilton", "Monza", sample_df)

def test_position_moyenne_par_GP():
    sample_data = {
        "Pilote": ["Hamilton", "Verstappen", "Hamilton", "Verstappen", "Hamilton"],
        "Position": [1, 2, 3, 4, 5],
        "GP": ["Australia", "Bahrain", "China", "Monaco", "Canada"],
        "Saison": [2023, 2023, 2023, 2023, 2023],
    }
    sample_df = pl.DataFrame(sample_data)

    moyenne = position_moyenne_par_GP("Hamilton", "Canada", sample_df)
    assert moyenne == 5.0, "La moyenne pour Hamilton au GP de Canada doit être 5.0."

    moyenne = position_moyenne_par_GP("Verstappen", "Bahrain", sample_df)
    assert moyenne == 2.0, "La moyenne pour Verstappen au GP de Bahrain doit être 2.0."

def test_plantage_position_moyenne_par_GP():
    sample_data = {
        "Pilote": ["Hamilton", "Verstappen", "Hamilton", "Verstappen", "Hamilton"],
        "Position": [1, 2, 3, 4, 5],
        "GP": ["Australia", "Bahrain", "China", "Monaco", "Canada"],
        "Saison": [2023, 2023, 2023, 2023, 2023],
    }
    sample_df = pl.DataFrame(sample_data)

    with pytest.raises(ValueError, match="Le pilote 'Leclerc' n'est pas présent.*"):
        position_moyenne_par_GP("Leclerc", "Canada", sample_df)

    with pytest.raises(ValueError, match="Le Grand Prix 'Monza' n'est pas disponible.*"):
        position_moyenne_par_GP("Hamilton", "Monza", sample_df)

    empty_df = pl.DataFrame({"Pilote": [], "Position": [], "GP": [], "Saison": []})
    with pytest.raises(ValueError, match="Le pilote 'Hamilton' n'est pas présent.*"):
        position_moyenne_par_GP("Hamilton", "Canada", empty_df)

def test_position_moyenne_par_saison():
    sample_data = {
        "Pilote": ["Hamilton", "Hamilton", "Verstappen", "Hamilton", "Verstappen"],
        "Position": [1, 2, 3, 4, 5],
        "Saison": [2021, 2022, 2021, 2022, 2022],
    }
    sample_df = pl.DataFrame(sample_data)

    fig = position_moyenne_par_saison("Hamilton", sample_df)
    assert isinstance(fig, go.Figure), "La fonction doit retourner un objet de type plotly.graph_objects.Figure."
    assert len(fig.data[0].x) == 2, "Le graphique doit inclure deux saisons pour Hamilton."
    assert len(fig.data[0].y) == 2, "Le graphique doit inclure les moyennes pour deux saisons."

def test_plantage_position_moyenne_par_saison():
    sample_data = {
        "Pilote": ["Hamilton", "Verstappen", "Hamilton", "Verstappen", "Hamilton"],
        "Position": [1, 2, 3, 4, 5],
        "Saison": [2021, 2021, 2022, 2022, 2022],
    }
    sample_df = pl.DataFrame(sample_data)

    with pytest.raises(ValueError, match="Le pilote 'Leclerc' n'est pas présent.*"):
        position_moyenne_par_saison("Leclerc", sample_df)

    empty_df = pl.DataFrame({"Pilote": [], "Position": [], "Saison": []})
    with pytest.raises(ValueError, match="Le pilote 'Hamilton' n'est pas présent.*"):
        position_moyenne_par_saison("Hamilton", empty_df)

def test_stats_pilotes_saisons_success():
    sample_data = {
        "Pilote": ["Hamilton", "Hamilton", "Verstappen", "Hamilton", "Verstappen"],
        "Position": [1, 2, 3, 4, 5],
        "Saison": [2021, 2022, 2021, 2022, 2022],
        "Pluie": [1, 0, 1, 0, 1],
        "Circuit_Lent": [1, 0, 0, 1, 1],
        "Circuit_Moderé": [0, 1, 1, 0, 0],
        "Circuit_Rapide": [0, 0, 0, 1, 1],
        "Vitesse_Moyenne": [200, 210, 205, 215, 220],
    }
    sample_df = pl.DataFrame(sample_data)

    stats = stats_pilotes_saisons("Hamilton", 2022, sample_df)
    assert stats == (3.0, None, 215.0, 210.0, 215), "Les statistiques pour Hamilton en 2022 doivent être correctes."

def test_plantage_stats_pilotes_saisons():
    sample_data = {
        "Pilote": ["Hamilton", "Verstappen", "Hamilton", "Verstappen", "Hamilton"],
        "Position": [1, 2, 3, 4, 5],
        "Saison": [2021, 2021, 2022, 2022, 2022],
        "Pluie": [1, 1, 0, 1, 0],
        "Circuit_Lent": [1, 0, 1, 0, 0],
        "Circuit_Moderé": [0, 1, 0, 1, 0],
        "Circuit_Rapide": [0, 0, 1, 0, 1],
        "Vitesse_Moyenne": [200, 210, 220, 230, 240],
    }
    sample_df = pl.DataFrame(sample_data)

    with pytest.raises(ValueError, match="Le pilote 'Leclerc' n'est pas présent.*"):
        stats_pilotes_saisons("Leclerc", 2022, sample_df)

    with pytest.raises(ValueError, match="La saison '2020' n'est pas disponible.*"):
        stats_pilotes_saisons("Hamilton", 2020, sample_df)

def test_stats_ecuries_saisons():
    sample_data = {
        "Chassis": ["Mercedes", "Mercedes", "Red Bull", "Mercedes", "Red Bull"],
        "Position": [1, 2, 3, 4, 5],
        "Saison": [2021, 2022, 2021, 2022, 2022],
        "Pluie": [1, 0, 1, 0, 1],
        "Circuit_Lent": [1, 0, 0, 1, 1],
        "Circuit_Moderé": [0, 1, 1, 0, 0],
        "Circuit_Rapide": [0, 0, 0, 1, 1],
        "Vitesse_Moyenne": [200, 210, 205, 215, 220],
    }
    sample_df = pl.DataFrame(sample_data)

    stats = stats_ecuries_saisons("Mercedes", 2022, sample_df)
    assert stats == (3.0, None, 215.0, 210.0, 215), "Les statistiques pour Mercedes en 2022 doivent être correctes."

def test_plantage_stats_ecuries_saisons():
    sample_data = {
        "Chassis": ["Mercedes", "Red Bull", "Mercedes", "Red Bull", "Mercedes"],
        "Position": [1, 2, 3, 4, 5],
        "Saison": [2021, 2021, 2022, 2022, 2022],
        "Pluie": [1, 1, 0, 1, 0],
        "Circuit_Lent": [1, 0, 1, 0, 0],
        "Circuit_Moderé": [0, 1, 0, 1, 0],
        "Circuit_Rapide": [0, 0, 1, 0, 1],
        "Vitesse_Moyenne": [200, 210, 220, 230, 240],
    }
    sample_df = pl.DataFrame(sample_data)

    with pytest.raises(ValueError, match="L'écurie 'Ferrari' n'est pas présente.*"):
        stats_ecuries_saisons("Ferrari", 2022, sample_df)

    with pytest.raises(ValueError, match="La saison '2020' n'est pas disponible.*"):
        stats_ecuries_saisons("Mercedes", 2020, sample_df)

def test_correlation_matrix():
    sample_data = {
        "A": [1, 2, 3, 4, 5],
        "B": [5, 4, 3, 2, 1],
        "C": [2.0, 3.0, 4.0, 5.0, 6.0],
        "D": ["x", "y", "z", "w", "v"],
    }
    sample_df = pl.DataFrame(sample_data)

    fig = correlation_matrix(sample_df)

    assert isinstance(fig, go.Figure), "La fonction doit retourner un objet de type plotly.graph_objects.Figure."

def test_plantage_correlation_matrix():
    sample_data = {
        "A": ["x", "y", "z", "w", "v"],
        "B": ["a", "b", "c", "d", "e"],
    }
    sample_df = pl.DataFrame(sample_data)

    with pytest.raises(ValueError, match="Aucune colonne numérique.*"):
        correlation_matrix(sample_df)

def test_importance_variable():
    X = pd.DataFrame({
        "ligne1": [1, 2, 3, 4, 5],
        "ligne2": [5, 4, 3, 2, 1],
        "categorie": ["A", "B", "A", "B", "A"]
    })
    y = [1, 2, 3, 4, 5]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["ligne1", "ligne2"]),
            ("cat", OneHotEncoder(), ["categorie"])
        ]
    )

    model = RandomForestRegressor(random_state=42)

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("regressor", model)
    ])

    pipeline.fit(X, y)

    fig = importance_variable(pipeline, top_n=2)

    assert isinstance(fig, go.Figure), "La fonction doit retourner un graphique Plotly."

def test_plantage_importance_variable():
    pipeline = Pipeline([
        ("regressor", LinearRegression())
    ])

    try:
        importance_variable(pipeline)
    except AttributeError as e:
        assert "Le modèle n'a pas d'attribut 'feature_importances_'" in str(e)

def test_preparation_data_prediction():
    sample_data = pd.DataFrame({
        'Pilote': ['Hamilton', 'Verstappen', 'Hamilton'],
        'GP': ['Monaco', 'Monaco', 'Monaco'],
        'Saison': [2021, 2021, 2020],
        'Circuit': ['Monaco', 'Monaco', 'Monaco'],
        'Virages': [19, 19, 19],
        'Longueur_km': [3.337, 3.337, 3.337],
        'Position': [1, 2, 1],
        'Temps_sec': [120.0, 121.0, 119.0]
    })

    pilotes = ['Hamilton', 'Verstappen']
    chassis = ['Mercedes', 'Red Bull']
    moteurs = ['Mercedes', 'Honda']
    df_result = preparation_data_prediction(
        df=sample_data,
        pilotes=pilotes,
        gp='Monaco',
        saison=2022,
        numero_gp=5,
        chassis=chassis,
        moteurs=moteurs,
        pluie=0,
        nuit=0,
        circuit='Monaco'
    )

    assert isinstance(df_result, pd.DataFrame), "Le résultat doit être un DataFrame pandas."
    assert len(df_result) == len(pilotes), "Le DataFrame doit contenir une ligne par pilote."
    assert 'Virages' in df_result.columns, "Le DataFrame doit contenir la colonne 'Virages'."

def test_plantage_preparation_data_prediction():
    sample_data = pd.DataFrame({
        'Pilote': ['Hamilton', 'Verstappen', 'Hamilton'],
        'GP': ['Monaco', 'Monaco', 'Monaco'],
        'Saison': [2021, 2021, 2020],
        'Circuit': ['Monaco', 'Monaco', 'Monaco'],
        'Virages': [19, 19, 19],
        'Longueur_km': [3.337, 3.337, 3.337],
        'Position': [1, 2, 1],
        'Temps_sec': [120.0, 121.0, 119.0]
    })

    pilotes = ['Hamilton', 'Verstappen']
    chassis = ['Mercedes']
    moteurs = ['Mercedes', 'Honda']

    try:
        preparation_data_prediction(
            df=sample_data,
            pilotes=pilotes,
            gp='Monaco',
            saison=2022,
            numero_gp=5,
            chassis=chassis,
            moteurs=moteurs,
            pluie=0,
            nuit=0,
            circuit='Monaco'
        )
    except ValueError as e:
        assert "Les longueurs des listes pilotes, chassis et moteurs doivent être égales." in str(e)

def test_prediction_grille():
    historical_data = pd.DataFrame({
        "Pilote": ["Hamilton", "Verstappen"],
        "GP": ["Monaco", "Monaco"],
        "Saison": [2024, 2024],
        "Numeros_gp": [7, 7],
        "Chassis": ["Mercedes", "Red Bull"],
        "Moteur_Ferrari": [0, 0],
        "Moteur_Honda": [0, 1],
        "Moteur_Mercedes": [1, 0],
        "Moteur_Renault": [0, 0],
        "Pluie": [0, 0],
        "Nuit": [0, 0],
        "Circuit": ["Monaco", "Monaco"],
        "Virages": [18, 18],
        "Longueur_km": [3.337, 3.337],
        "Position": [6, 7],
        "Temps_sec": [170.238, 70.567],
        "position_gp_saison_precedente": [6, 7],
        "chrono_gp_saison_precedente": [70.567, 70.621]
    })

    pilotes = ["Hamilton", "Verstappen"]
    gp = "Monaco"
    saison = 2025
    numero_gp = 8
    chassis = ["Mercedes", "Red Bull"]
    moteurs = ["Mercedes", "Honda"]
    pluie = 0
    nuit = 0
    circuit = "Monaco"

    prepared_df = preparation_data_prediction(
        df=historical_data,
        pilotes=pilotes,
        gp=gp,
        saison=saison,
        numero_gp=numero_gp,
        chassis=chassis,
        moteurs=moteurs,
        pluie=pluie,
        nuit=nuit,
        circuit=circuit
    )

    model = joblib.load("data/ml/models/model_boosting_allure.pkl")

    result_df = prediction_grille(prepared_df, "data/ml/models/model_boosting_allure.pkl")

    assert isinstance(result_df, pd.DataFrame), "Le résultat doit être un DataFrame pandas."
    assert "Prediction" in result_df.columns, "Le DataFrame doit contenir une colonne 'Prediction'."
    assert len(result_df) == len(prepared_df), "Le DataFrame doit contenir autant de lignes que le DataFrame préparé."
    assert set(result_df.columns) == {"Pilote", "GP", "Saison", "Prediction"}, \
        "Le DataFrame doit contenir uniquement les colonnes 'Pilote', 'GP', 'Saison', et 'Prediction'."

def test_plantage_prediction_grille():
    empty_df = pd.DataFrame()
    try:
        prediction_grille(empty_df, "data/ml/models/model_boosting_allure.pkl")
    except RuntimeError as e:
        assert "Le DataFrame fourni est vide" in str(e), "Erreur non détectée pour un DataFrame vide"

    valid_df = pd.DataFrame({
        "feature1": [1.0, 2.0],
        "feature2": [3.0, 4.0],
        "Pilote": ["Hamilton", "Verstappen"],
        "GP": ["Monaco", "Monaco"],
        "Saison": [2025, 2025]
    })
    try:
        prediction_grille(valid_df, "invalid_model_path.pkl")
    except FileNotFoundError as e:
        assert "Le fichier du modèle spécifié" in str(e), "Erreur non détectée pour un modèle inexistant"

def test_ordonner_prediction():
    df_position = pd.DataFrame({
        "Pilote": ["Hamilton", "Verstappen", "Leclerc"],
        "GP": ["Monaco", "Monaco", "Monaco"],
        "Saison": [2025, 2025, 2025],
        "Prediction": [2, 1, 3]
    })

    result_position = ordonner_prediction(df_position, target="Allure")
    assert result_position is not None, "La fonction a échoué pour une cible valide ('Allure')."
    assert "Prediction Allure" in result_position.columns, "La colonne 'Prediction' n'a pas été renommée correctement pour 'Allure'."
    assert list(result_position["Pilote"]) == ["Verstappen", "Hamilton", "Leclerc"], \
        "Le tri par ordre croissant pour 'Allure' n'a pas fonctionné correctement."

def test_plantage_ordonner_prediction():
    df = pd.DataFrame({
        "Pilote": ["Hamilton", "Verstappen"],
        "GP": ["Monaco", "Monaco"],
        "Saison": [2025, 2025],
        "Prediction": [1, 2]
    })

    try:
        ordonner_prediction(df, target="Invalide")
    except Exception as e:
        assert "La cible doit être 'Position' ou 'Temps_sec'" in str(e), "Erreur non détectée pour une cible invalide."

    empty_df = pd.DataFrame()
    try:
        ordonner_prediction(empty_df, target="Position")
    except Exception as e:
        assert "Le DataFrame est vide" in str(e), "Erreur non détectée pour un DataFrame vide."

    df_missing_prediction = pd.DataFrame({
        "Pilote": ["Hamilton", "Verstappen"],
        "GP": ["Monaco", "Monaco"],
        "Saison": [2025, 2025]
    })

    try:
        ordonner_prediction(df_missing_prediction, target="Position")
    except Exception as e:
        assert "Prediction" in str(e), "Erreur non détectée pour une colonne 'Prediction' manquante."