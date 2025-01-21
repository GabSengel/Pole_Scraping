import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import pandas as pd
import streamlit as st
import joblib

@st.cache_data
def load_data():
    return pl.read_json("data/clean_df/DataFrame.json")

def load_data_metrics(modele : str):
    if modele == "boosting_Position" :
        return pl.read_csv("data/ml/metrics/metrics_boosting_Position.csv")
    elif modele == "boosting_Temps_sec" :
        return pl.read_csv("data/ml/metrics/metrics_boosting_Temps_sec.csv")
    elif modele == "rf_Position" : 
        return pl.read_csv("data/ml/metrics/metrics_rf_Position.csv")
    elif modele == "rf_Temps_sec" :
        return pl.read_csv("data/ml/metrics/metrics_rf_Temps_sec.csv")
    elif modele == "boosting_allure" :
        return pl.read_csv("data/ml/metrics/metrics_boosting_allure.csv")

def load_data_resultats_modeles(modele : str):
    if modele == "boosting_Position" :
        return pl.read_csv("data/ml/results/results_boosting_Position.csv")
    elif modele == "boosting_Temps_sec" :
        return pl.read_csv("data/ml/results/results_boosting_Temps_sec.csv")
    elif modele == "rf_Position" : 
        return pl.read_csv("data/ml/results/results_rf_Position.csv")
    elif modele == "rf_Temps_sec" :
        return pl.read_csv("data/ml/results/results_rf_Temps_sec.csv")
    elif modele == "boosting_allure" :
        return pl.read_csv("data/ml/results/results_boosting_allure.csv")

def load_data_param_modeles(modele : str):
    if modele == "boosting_Position" :
        return pl.read_csv("data/ml/parameters/param_boosting_Position.csv")
    elif modele == "boosting_Temps_sec" :
        return pl.read_csv("data/ml/parameters/param_boosting_Temps_sec.csv")
    elif modele == "rf_Position" : 
        return pl.read_csv("data/ml/parameters/param_rf_Position.csv")
    elif modele == "rf_Temps_sec" :
        return pl.read_csv("data/ml/parameters/param_rf_Temps_sec.csv")
    elif modele == "boosting_allure" :
        return pl.read_csv("data/ml/parameters/param_boosting_allure.csv")

def load_modele(modele : str):
    if modele == "boosting_Position" :
        return joblib.load('data/ml/models/model_boosting_Position.pkl')
    elif modele == "boosting_Temps_sec" :
        return joblib.load('data/ml/models/model_boosting_Temps_sec.pkl')
    elif modele == "rf_Position" : 
        return joblib.load('data/ml/models/model_rf_Position.pkl')
    elif modele == "rf_Temps_sec" :
        return joblib.load('data/ml/models/model_rf_Temps_sec.pkl')
    elif modele == "boosting_allure" :
        return joblib.load('data/ml/models/model_boosting_allure.pkl')

def historic_positions(Pilote: str, nombres_gp: int, df):
    """
    Génère une visualisation de l'évolution des positions de départ d'un pilote 
    sur ses derniers Grands Prix.

    Args:
        Pilote (str): Nom du pilote pour lequel la visualisation est générée.
        nombres_gp (int): Nombre de Grands Prix les plus récents à inclure dans l'analyse.
        df (polars.DataFrame): DataFrame contenant les données, avec au moins les colonnes 
            'Pilote', 'Position', 'GP', et 'Saison'.

    Returns:
        plotly.graph_objects.Figure: Un objet Plotly représentant une ligne montrant 
        l'évolution des positions de départ du pilote spécifié.

    Raises:
        ValueError: Si le pilote spécifié n'est pas présent dans le DataFrame.
        ValueError: Si le nombre de Grands Prix spécifié est supérieur au nombre 
        disponible pour le pilote dans le DataFrame.
    """
    if Pilote not in df.select(pl.col("Pilote")).unique().to_series().to_list():
        raise ValueError(f"Le pilote '{Pilote}' n'est pas présent dans les données fournies.")

    df2 = df.filter(pl.col("Pilote") == Pilote)

    if nombres_gp > len(df2):
        raise ValueError(
            f"Le nombre de Grands Prix spécifié ({nombres_gp}) est supérieur au "
            f"nombre de données disponibles ({len(df2)}) pour le pilote {Pilote}."
        )

    position = df2.select(pl.col("Position")).to_series().to_list()[-nombres_gp:]
    gp = df2.select(pl.col("GP")).to_series().to_list()[-nombres_gp:]
    saison = df2.select(pl.col("Saison")).to_series().to_list()[-nombres_gp:]

    x_labels = [f"{gp[i]} {saison[i]}" for i in range(len(gp))]

    fig = px.line(
        x=x_labels,
        y=position,
        title=f"Evolution des positions sur la grille pour {Pilote} sur ses {nombres_gp} derniers GP",
        labels={"x": "GP", "y": "Position"}
    )

    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(showticklabels=False)

    return fig

def position_moyenne(Pilote: str, nombres_gp: int, df):
    """
    Calcule la position moyenne de départ d'un pilote sur ses derniers Grands Prix.

    Args:
        Pilote (str): Nom du pilote pour lequel la moyenne est calculée.
        nombres_gp (int): Nombre de Grands Prix les plus récents à inclure dans le calcul.
        df (polars.DataFrame): DataFrame contenant les données, avec au moins les colonnes 
            'Pilote' et 'Position'.

    Returns:
        float: La position moyenne arrondie à deux décimales.

    Raises:
        ValueError: Si le pilote spécifié n'est pas présent dans le DataFrame.
        ValueError: Si le nombre de Grands Prix spécifié est supérieur au nombre 
        disponible pour le pilote dans le DataFrame.
        ZeroDivisionError: Si aucune donnée n'est disponible pour calculer la moyenne.
    """
    if Pilote not in df.select(pl.col("Pilote")).unique().to_series().to_list():
        raise ValueError(f"Le pilote '{Pilote}' n'est pas présent dans les données fournies.")

    df2 = df.filter(pl.col("Pilote") == Pilote)

    if nombres_gp > len(df2):
        raise ValueError(
            f"Le nombre de Grands Prix spécifié ({nombres_gp}) est supérieur au "
            f"nombre de données disponibles ({len(df2)}) pour le pilote {Pilote}."
        )

    position = df2.select(pl.col("Position")).to_series().to_list()[-nombres_gp:]

    if len(position) == 0:
        raise ZeroDivisionError("Aucune donnée disponible pour calculer la moyenne.")

    moyenne = sum(position) / len(position)
    return round(moyenne, 2)


def historic_positions_par_gp(Pilote: str, GP: str, df):
    """
    Génère une visualisation de l'évolution des positions de départ d'un pilote 
    pour un Grand Prix spécifique.

    Args:
        Pilote (str): Nom du pilote pour lequel la visualisation est générée.
        GP (str): Nom du Grand Prix pour lequel les données sont filtrées.
        df (polars.DataFrame): DataFrame contenant les données, avec au moins les colonnes 
            'Pilote', 'Position', 'GP', et 'Saison'.

    Returns:
        plotly.graph_objects.Figure: Un graphique Plotly représentant les positions 
        de départ du pilote pour le Grand Prix spécifié.

    Raises:
        ValueError: Si le pilote spécifié n'est pas présent dans le DataFrame.
        ValueError: Si le Grand Prix spécifié n'est pas disponible pour le pilote 
        dans les données filtrées.
    """
    if Pilote not in df.select(pl.col("Pilote")).unique().to_series().to_list():
        raise ValueError(f"Le pilote '{Pilote}' n'est pas présent dans les données fournies.")

    df2 = df.filter(pl.col("Pilote") == Pilote)

    if GP not in df2.select(pl.col("GP")).unique().to_series().to_list():
        raise ValueError(
            f"Le Grand Prix '{GP}' n'est pas disponible pour le pilote '{Pilote}' dans les données fournies."
        )

    df2 = df2.filter(pl.col("GP") == GP)

    position = df2.select(pl.col("Position")).to_series().to_list()
    gp = df2.select(pl.col("GP")).to_series().to_list()
    saison = df2.select(pl.col("Saison")).to_series().to_list()

    x_labels = [f"{gp[i]} {saison[i]}" for i in range(len(gp))]

    fig = px.line(
        x=x_labels,
        y=position,
        title=f"Evolution des positions sur la grille pour {Pilote} au GP de {GP}",
        labels={"x": "GP", "y": "Position"}
    )

    fig.update_yaxes(autorange="reversed")

    return fig

def position_moyenne_par_GP(Pilote: str, GP: str, df):
    """
    Calcule la position moyenne de départ d'un pilote pour un Grand Prix spécifique.

    Args:
        Pilote (str): Nom du pilote pour lequel la moyenne est calculée.
        GP (str): Nom du Grand Prix pour lequel les données sont filtrées.
        df (polars.DataFrame): DataFrame contenant les données, avec au moins les colonnes 
            'Pilote', 'Position', et 'GP'.

    Returns:
        float: La position moyenne arrondie à deux décimales.

    Raises:
        ValueError: Si le pilote spécifié n'est pas présent dans le DataFrame.
        ValueError: Si le Grand Prix spécifié n'est pas disponible pour le pilote 
        dans les données filtrées.
        ZeroDivisionError: Si aucune donnée n'est disponible pour calculer la moyenne.
    """
    if Pilote not in df.select(pl.col("Pilote")).unique().to_series().to_list():
        raise ValueError(f"Le pilote '{Pilote}' n'est pas présent dans les données fournies.")

    df2 = df.filter(pl.col("Pilote") == Pilote)

    if GP not in df2.select(pl.col("GP")).unique().to_series().to_list():
        raise ValueError(
            f"Le Grand Prix '{GP}' n'est pas disponible pour le pilote '{Pilote}' dans les données fournies."
        )

    df2 = df2.filter(pl.col("GP") == GP)

    position = df2.select(pl.col("Position")).to_series().to_list()

    if len(position) == 0:
        raise ZeroDivisionError("Aucune donnée disponible pour calculer la moyenne.")

    moyenne = sum(position) / len(position)
    return round(moyenne, 2)


def position_moyenne_par_saison(Pilote: str, df):
    """
    Génère une visualisation de l'évolution de la position moyenne de départ d'un pilote 
    sur toutes ses saisons de F1.

    Args:
        Pilote (str): Nom du pilote pour lequel la visualisation est générée.
        df (polars.DataFrame): DataFrame contenant les données, avec des colonnes 
            'Pilote', 'Position', et 'Saison'.

    Returns:
        plotly.graph_objects.Figure: Un graphique Plotly représentant l'évolution de la 
        position moyenne par saison pour le pilote spécifié.

    Raises:
        ValueError: Si le pilote spécifié n'est pas présent dans le DataFrame.
        ValueError: Si aucune donnée n'est disponible pour le pilote après filtrage.
    """
    if Pilote not in df.select(pl.col("Pilote")).unique().to_series().to_list():
        raise ValueError(f"Le pilote '{Pilote}' n'est pas présent dans les données fournies.")

    df2 = (
        df.filter(pl.col("Pilote") == Pilote)
        .group_by("Saison")
        .agg([pl.col("Position").mean().round(2).alias("Position moyenne")])
        .sort("Saison")
    )

    if len(df2) == 0:
        raise ValueError(f"Aucune donnée n'est disponible pour le pilote '{Pilote}' après le filtrage.")

    fig = px.line(
        df2,
        x="Saison",
        y="Position moyenne",
        title=f"Position moyenne de {Pilote} sur toutes ses saisons de F1",
        labels={"value": "Moyenne", "Saison": "Saison"},
        markers=True
    )

    fig.update_yaxes(title="Position moyenne", autorange="reversed")
    fig.update_xaxes(title="Saison")

    return fig

def stats_pilotes_saisons(Pilote: str, Saison: int, df):
    """
    Calcule des statistiques spécifiques pour un pilote sur une saison donnée.

    Args:
        Pilote (str): Nom du pilote pour lequel les statistiques sont calculées.
        Saison (int): Année de la saison pour laquelle les statistiques sont calculées.
        df (polars.DataFrame): DataFrame contenant les données, avec au moins les colonnes
            'Pilote', 'Position', 'Saison', 'Pluie', 'Circuit_Lent', 'Circuit_Moderé',
            'Circuit_Rapide', et 'Vitesse_Moyenne'.

    Returns:
        tuple: Un tuple contenant :
            - position_moyenne (float) : Position moyenne du pilote sur la saison.
            - position_moyenne_pluie (float or None) : Position moyenne du pilote sur les circuits pluvieux.
            - vitesse_moyenne_lent (float or None) : Vitesse moyenne du pilote sur les circuits lents.
            - vitesse_moyenne_modere (float or None) : Vitesse moyenne du pilote sur les circuits modérés.
            - vitesse_moyenne_rapide (float or None) : Vitesse moyenne du pilote sur les circuits rapides.

    Raises:
        ValueError: Si le pilote spécifié n'est pas présent dans le DataFrame.
        ValueError: Si la saison spécifiée n'est pas disponible pour le pilote dans les données filtrées.
        ZeroDivisionError: Si aucune donnée n'est disponible pour calculer les statistiques.
    """
    if Pilote not in df.select(pl.col("Pilote")).unique().to_series().to_list():
        raise ValueError(f"Le pilote '{Pilote}' n'est pas présent dans les données fournies.")

    df2 = df.filter(pl.col("Pilote") == Pilote)

    if Saison not in df2.select(pl.col("Saison")).unique().to_series().to_list():
        raise ValueError(
            f"La saison '{Saison}' n'est pas disponible pour le pilote '{Pilote}' dans les données fournies."
        )

    df2 = df2.filter(pl.col("Saison") == Saison)

    def safe_mean(filter_df, col_name):
        """Pour gérer les None."""
        if len(filter_df) == 0:
            return None
        result = filter_df.select(pl.col(col_name).mean()).item()
        return round(result, 2) if result is not None else None

    position_moyenne = safe_mean(df2, "Position")
    position_moyenne_pluie = safe_mean(df2.filter(pl.col("Pluie") == 1), "Position")
    vitesse_moyenne_lent = safe_mean(df2.filter(pl.col("Circuit_Lent") == 1), "Vitesse_Moyenne")
    vitesse_moyenne_modere = safe_mean(df2.filter(pl.col("Circuit_Moderé") == 1), "Vitesse_Moyenne")
    vitesse_moyenne_rapide = safe_mean(df2.filter(pl.col("Circuit_Rapide") == 1), "Vitesse_Moyenne")

    return position_moyenne, position_moyenne_pluie, vitesse_moyenne_lent, vitesse_moyenne_modere, vitesse_moyenne_rapide

def stats_ecuries_saisons(Ecurie: str, Saison: int, df):
    """
    Calcule des statistiques spécifiques pour une écurie sur une saison donnée.

    Args:
        Ecurie (str): Nom de l'écurie pour laquelle les statistiques sont calculées.
        Saison (int): Année de la saison pour laquelle les statistiques sont calculées.
        df (polars.DataFrame): DataFrame contenant les données, avec au moins les colonnes
            'Chassis', 'Position', 'Saison', 'Pluie', 'Circuit_Lent', 'Circuit_Moderé',
            'Circuit_Rapide', et 'Vitesse_Moyenne'.

    Returns:
        tuple: Un tuple contenant :
            - position_moyenne (float) : Position moyenne de l'écurie sur la saison.
            - position_moyenne_pluie (float or None) : Position moyenne de l'écurie sur les circuits pluvieux.
            - vitesse_moyenne_lent (float or None) : Vitesse moyenne de l'écurie sur les circuits lents.
            - vitesse_moyenne_modere (float or None) : Vitesse moyenne de l'écurie sur les circuits modérés.
            - vitesse_moyenne_rapide (float or None) : Vitesse moyenne de l'écurie sur les circuits rapides.

    Raises:
        ValueError: Si l'écurie spécifiée n'est pas présente dans le DataFrame.
        ValueError: Si la saison spécifiée n'est pas disponible pour l'écurie dans les données filtrées.
        ZeroDivisionError: Si aucune donnée n'est disponible pour calculer les statistiques.
    """
    if Ecurie not in df.select(pl.col("Chassis")).unique().to_series().to_list():
        raise ValueError(f"L'écurie '{Ecurie}' n'est pas présente dans les données fournies.")

    df2 = df.filter(pl.col("Chassis") == Ecurie)

    if Saison not in df2.select(pl.col("Saison")).unique().to_series().to_list():
        raise ValueError(
            f"La saison '{Saison}' n'est pas disponible pour l'écurie '{Ecurie}' dans les données fournies."
        )

    df2 = df2.filter(pl.col("Saison") == Saison)

    def safe_mean(filter_df, col_name):
        """Pour gérer les None."""
        if len(filter_df) == 0:
            return None
        result = filter_df.select(pl.col(col_name).mean()).item()
        return round(result, 2) if result is not None else None

    position_moyenne = safe_mean(df2, "Position")
    position_moyenne_pluie = safe_mean(df2.filter(pl.col("Pluie") == 1), "Position")
    vitesse_moyenne_lent = safe_mean(df2.filter(pl.col("Circuit_Lent") == 1), "Vitesse_Moyenne")
    vitesse_moyenne_modere = safe_mean(df2.filter(pl.col("Circuit_Moderé") == 1), "Vitesse_Moyenne")
    vitesse_moyenne_rapide = safe_mean(df2.filter(pl.col("Circuit_Rapide") == 1), "Vitesse_Moyenne")

    return position_moyenne, position_moyenne_pluie, vitesse_moyenne_lent, vitesse_moyenne_modere, vitesse_moyenne_rapide

def correlation_matrix(df):
    """
    Génère une matrice de corrélation des colonnes numériques dans un DataFrame.

    Args:
        df (polars.DataFrame): DataFrame contenant les données. Les colonnes numériques doivent 
            être de type `pl.Float64` ou `pl.Int64`.

    Returns:
        plotly.graph_objects.Figure: Un graphique Plotly représentant la matrice de corrélation.

    Raises:
        ValueError: Si aucune colonne numérique n'est présente dans le DataFrame.
    """
    numeric_cols = [col for col, dtype in zip(df.columns, df.schema.values()) if dtype in (pl.Float64, pl.Int64)]

    if not numeric_cols:
        raise ValueError("Aucune colonne numérique (pl.Float64 ou pl.Int64) n'est présente dans le DataFrame.")

    df_numeric = df.select(numeric_cols)

    df_numeric_pd = df_numeric.to_pandas()
    correlation_matrix = df_numeric_pd.corr()

    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        color_continuous_scale="RdBu",
        title="Matrice de corrélation des variables numériques"
    )
    fig.update_layout(
        autosize=False,
        width=1200,
        height=800,
        title_x=0.5,
    )

    return fig

def importance_variable(best_pipeline, top_n=None, min_importance=None):
    """
    Crée un graphique montrant l'importance des variables à partir d'un modèle entraîné.

    Args:
        best_pipeline (Pipeline): Pipeline entraîné contenant un modèle (avec `feature_importances_`) 
            et un préprocesseur (`preprocessing`).
        top_n (int, optional): Nombre maximum de variables les plus importantes à afficher.
            Par défaut, affiche toutes les variables.
        min_importance (float, optional): Seuil minimal d'importance pour inclure une variable
            dans le tableau. Par défaut, aucune limite.

    Raises:
        AttributeError: Si le modèle dans le pipeline n'a pas l'attribut `feature_importances_`.
        ValueError: Si le nombre de colonnes après transformation ne correspond pas au nombre
            de valeurs dans `feature_importances_`.

    Returns:
        plotly.graph_objects.Figure: Un graphique Plotly en barres horizontales montrant
        les importances des variables.
    """
    best_model = best_pipeline.named_steps["regressor"]
    
    if not hasattr(best_model, "feature_importances_"):
        raise AttributeError("Le modèle n'a pas d'attribut 'feature_importances_'.")
    
    feature_importances = best_model.feature_importances_
    
    preprocessor = best_pipeline.named_steps["preprocessing"]
    transformed_columns = []
    
    for name, transformer, columns in preprocessor.transformers_:
        if transformer == "drop" or transformer is None:
            continue
        elif hasattr(transformer, "get_feature_names_out"):
            transformed_columns.extend(
                transformer.get_feature_names_out(columns)
            )
        else:
            transformed_columns.extend(columns)

    if len(feature_importances) != len(transformed_columns):
        raise ValueError(
            "Le nombre de colonnes après transformation ne correspond pas au nombre de valeurs dans 'feature_importances_'."
        )

    importance_df = pd.DataFrame({
        "Feature": transformed_columns,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False)

    if min_importance is not None:
        importance_df = importance_df[importance_df["Importance"] >= min_importance]
    if top_n is not None:
        importance_df = importance_df.head(top_n)
        
    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Importance des Variables (faites glisser votre souris sur les barres pour afficher le nom des variables)",
        labels={"Feature": "Variable", "Importance": "Importance"},
        template="plotly_white"
    )
        
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    return fig

def preparation_data_prediction(
    df: pd.DataFrame,
    pilotes: list,
    gp: str,
    saison: int,
    numero_gp: int,
    chassis: list,
    moteurs: list,
    pluie: int,
    nuit: int,
    circuit: str
) -> pd.DataFrame:
    """
    Prépare les données pour effectuer des prédictions sur un Grand Prix donné.

    Args:
        df (pd.DataFrame): DataFrame contenant les données historiques des Grands Prix.
        pilotes (list): Liste des noms des pilotes participant au Grand Prix.
        gp (str): Nom du Grand Prix.
        saison (int): Année de la saison en cours.
        numero_gp (int): Numéro du Grand Prix dans la saison.
        chassis (list): Liste des châssis associés aux pilotes (même longueur que `pilotes`).
        moteurs (list): Liste des moteurs associés aux pilotes (même longueur que `pilotes`).
        pluie (int): Indicateur binaire pour les conditions de pluie (0 ou 1).
        nuit (int): Indicateur binaire pour les conditions de nuit (0 ou 1).
        circuit (str): Nom du circuit sur lequel le Grand Prix est organisé.

    Raises:
        ValueError: Si les longueurs des listes `pilotes`, `chassis` et `moteurs` ne sont pas identiques.

    Returns:
        pd.DataFrame: DataFrame contenant les données formatées et prêtes pour la prédiction.
    """
    if not (len(pilotes) == len(chassis) == len(moteurs)):
        raise ValueError("Les longueurs des listes pilotes, chassis et moteurs doivent être égales.")

    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    circuit_data = df[df['Circuit'] == circuit].iloc[0] if circuit in df['Circuit'].tolist() else None
    nombre_virages = circuit_data['Virages'] if circuit_data is not None else -1
    longueur_km = circuit_data['Longueur_km'] if circuit_data is not None else -1

    prediction_data = []

    for i, pilote in enumerate(pilotes):
        saison_precedente = saison - 1
        pilote_gp_data = df[(df['Pilote'] == pilote) & (df['GP'] == gp) & (df['Saison'] == saison_precedente)]

        position_gp_saison_precedente = (
            pilote_gp_data['Position'].iloc[0] if not pilote_gp_data.empty else -1
        )
        chrono_gp_saison_precedente = (
            pilote_gp_data['Temps_sec'].iloc[0] if not pilote_gp_data.empty else -1
        )

        prediction_data.append({
            'Pilote': pilote,
            'GP': gp,
            'Saison': saison,
            'Numeros_gp': numero_gp,
            'Chassis': chassis[i],
            'Moteur_Ferrari': 1 if moteurs[i] == 'Ferrari' else 0,
            'Moteur_Honda': 1 if moteurs[i] == 'Honda' else 0,
            'Moteur_Mercedes': 1 if moteurs[i] == 'Mercedes' else 0,
            'Moteur_Renault': 1 if moteurs[i] == 'Renault' else 0,
            'Pluie': pluie,
            'Nuit': nuit,
            'Circuit': circuit,
            'Virages': nombre_virages,
            'Longueur_km': longueur_km,
            'position_gp_saison_precedente': position_gp_saison_precedente,
            'chrono_gp_saison_precedente': chrono_gp_saison_precedente
        })

    return pd.DataFrame(prediction_data)

def prediction_grille(prepared_df: pd.DataFrame, model_path: str) -> pd.DataFrame:
    """
    Prédit les positions pour un Grand Prix futur à partir d'un modèle pré-entraîné.

    Args:
        prepared_df (pd.DataFrame): DataFrame contenant les données préparées pour la prédiction.
            Les colonnes doivent correspondre aux caractéristiques attendues par le modèle.
        model_path (str): Chemin vers le fichier du modèle pré-entraîné (au format .joblib).

    Returns:
        pd.DataFrame: DataFrame avec les colonnes "Pilote", "GP", "Saison" et "Prediction",
        représentant les résultats prédits pour chaque pilote.

    Raises:
        FileNotFoundError: Si le fichier du modèle n'est pas trouvé.
        ValueError: Si les colonnes de `prepared_df` ne correspondent pas aux caractéristiques attendues par le modèle.
        ValueError: Si le DataFrame fourni est vide.
        Exception: Pour toute autre erreur survenant lors du chargement du modèle ou de la prédiction.
    """
    try:
        if prepared_df.empty:
            raise ValueError("Le DataFrame fourni est vide et ne peut pas être utilisé pour la prédiction.")
        
        model = joblib.load(model_path)

        expected_features = model.feature_names_in_
        for col in expected_features:
            if col not in prepared_df.columns:
                prepared_df[col] = 0

        prepared_df = prepared_df[expected_features]

        predictions = model.predict(prepared_df)

        prepared_df = prepared_df.copy()
        prepared_df.loc[:, "Prediction"] = predictions

        return prepared_df[["Pilote", "GP", "Saison", "Prediction"]]

    except FileNotFoundError:
        raise FileNotFoundError(f"Le fichier du modèle spécifié à '{model_path}' est introuvable.")
    except AttributeError:
        raise ValueError("Le modèle chargé ne contient pas l'attribut 'feature_names_in_' ou est incompatible.")
    except Exception as e:
        raise RuntimeError(f"Une erreur s'est produite lors de la prédiction : {e}")


def ordonner_prediction(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Modifie le nom de la colonne "Prediction" en fonction de la cible et trie le DataFrame par ordre croissant des prédictions.

    Args:
        df (pd.DataFrame): DataFrame contenant les résultats de prédiction.
        target (str): Le nom de la variable cible (Dans notre cas "Allure").

    Returns:
        pd.DataFrame: DataFrame modifié avec une colonne renommée et trié.
    """
    try:
        if target == "Allure":
            df = df.rename(columns={"Prediction": "Prediction Allure"})
        else:
            raise ValueError("La cible doit être 'Allure'.")

        df = df.sort_values(by=df.columns[-1], ascending=True)

        return df

    except Exception as e:
        print(f"Une erreur s'est produite lors du formatage et du tri : {e}")
        return None
