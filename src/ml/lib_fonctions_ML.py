from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import joblib
import json

def modif_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute une colonne 'Allure' au DataFrame en divisant 'Temps_sec' par 'Longueur_km'.
    
    Args:
        data (pd.DataFrame): Le DataFrame contenant les colonnes 'Temps_sec' et 'Longueur_km'.
        
    Returns:
        pd.DataFrame: Le DataFrame modifié avec la colonne 'Allure' ajoutée.
    """

    if 'Temps_sec' not in data.columns or 'Longueur_km' not in data.columns:
        raise ValueError("Les colonnes 'Temps_sec' et 'Longueur_km' doivent être présentes dans le DataFrame.")
    
    data['Allure'] = data['Temps_sec'] / data['Longueur_km']
    
    return data

def split_train_test(data: pd.DataFrame, variable: str):
    """
    Divise les données en ensembles d'entraînement et de test.
    
    Args:
        data (pd.DataFrame): Le DataFrame contenant les données.
        variable (str): Le nom de la colonne cible (variable à prédire).
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
            - X_train (pd.DataFrame): Caractéristiques pour l'entraînement.
            - X_test (pd.DataFrame): Caractéristiques pour le test.
            - y_train (pd.Series): Cible pour l'entraînement.
            - y_test (pd.Series): Cible pour le test.
    """
    X = data.drop(columns=[variable])
    y = data[variable]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=60)
    
    return X_train, X_test, y_train, y_test

def preprocess_data() -> ColumnTransformer:
    categorical_columns = ['Pilote', 'GP', 'Chassis', 'Circuit']
    numerical_columns = ['Saison', 'Numeros_gp', 'Moteur_Ferrari', 'Moteur_Honda', 
                    'Moteur_Mercedes', 'Moteur_Renault', 'Pluie', 'Nuit', 
                    'Circuit_Lent', 'Circuit_Moderé', 'Circuit_Rapide', 
                    'Virages', 'position_gp_saison_precedente', 
                    'chrono_gp_saison_precedente']
    
    preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_columns),
        ('num', StandardScaler(), numerical_columns)
    ],
    remainder='drop')

    
    return preprocessor

def boosting_entrainement(X_train, y_train, preprocessor):
    """
    Entraîne un modèle de Gradient Boosting avec optimisation des hyperparamètres via GridSearchCV.

    Args:
        X_train (pd.DataFrame): Les données d'entraînement (variables explicatives).
        y_train (pd.Series): Les cibles associées aux données d'entraînement.
        preprocessor (ColumnTransformer): Un objet de prétraitement des données pour traiter 
            les colonnes catégoriques et numériques.
    
    Returns:
        tuple: (Pipeline, pd.DataFrame)
            - best_pipeline (Pipeline): Le pipeline contenant le préprocesseur et le modèle 
            optimisé avec les meilleurs hyperparamètres.
            - best_param_df (pd.DataFrame): Un DataFrame contenant les meilleurs hyperparamètres 
            trouvés par GridSearchCV.
    """
    model = GradientBoostingRegressor(random_state=60)
    
    pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('regressor', model)
    ])
    
    param_grid = {
        'regressor__n_estimators': [100, 300, 500, 1000, 1500, 2000],
        'regressor__learning_rate': [0.01, 0.05, 0.1],
        'regressor__max_depth': [1, 3, 5, 7]
    }
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='neg_mean_absolute_error',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    result = grid_search.fit(X_train, y_train)
    best_pipeline = result.best_estimator_
    best_param_df = pd.DataFrame(grid_search.best_params_, index=[0])
    
    
    return best_pipeline, best_param_df


def boosting_test(best_pipeline, X_test, y_test):
    """
    Évalue un modèle de Boosting sur un ensemble de test et retourne les métriques et les résultats détaillés.
    
    Args:
        best_pipeline (Pipeline): Pipeline optimisé contenant le modèle entraîné.
        X_test (pd.DataFrame): Données de test (caractéristiques).
        y_test (pd.Series): Données de test (cible réelle).
    
    Returns:
        tuple: (metrics_df, results_df)
            - metrics_df (pd.DataFrame): Tableau des métriques d'évaluation (RMSE, MAE, R²).
            - results_df (pd.DataFrame): DataFrame contenant les valeurs réelles et prédictions.
    """

    y_pred = best_pipeline.predict(X_test)
    

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    

    metrics_df = pd.DataFrame({
        "Metric": ["RMSE", "MAE", "R²"],
        "Value": [rmse, mae, r2]
    })
    

    results_df = X_test.copy()
    results_df["Réalité"] = y_test
    results_df["Prédiction"] = y_pred
    results_df = results_df[["Pilote", "GP", "Saison", "Réalité", "Prédiction"]]
    
    return metrics_df, results_df


if __name__ == "__main__":

    data = pd.read_json("DataFrame.json")
    data=modif_data(data)

    X_train, X_test, y_train, y_test= split_train_test(data, 'Allure')
    preprocessor = preprocess_data()
    best_pipeline, param_boosting = boosting_entrainement(X_train, y_train, preprocessor)
    metrics_boosting, results_boosting = boosting_test(best_pipeline, X_test, y_test)


    results_boosting.to_csv(f"results_boosting_.csv", index=False)
    metrics_boosting.to_csv(f"metrics_boosting_.csv", index=False)
    param_boosting.to_csv(f"param_boosting_.csv", index=False)

    joblib.dump(best_pipeline, f"model_boosting_.pkl")