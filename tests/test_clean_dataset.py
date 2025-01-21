import pytest
import polars as pl
from src.clean_dataset import nettoyage_dataset

def test_nettoyage_dataset():
    df = nettoyage_dataset(data_set_circuit="data/results_scrap/data_set_infos_circuits.json", data_set_qualifs="data/results_scrap/data_set_qualifs.json")

    # Debug : Afficher les colonnes générées
    print("Colonnes générées :", df.columns)

    assert df.columns == ["Pilote", "GP", "Saison", "Numeros_gp", "Chassis", "Moteur_Ferrari", "Moteur_Honda", "Moteur_Mercedes", "Moteur_Renault", "Position",
                        "Temps_sec", "Vitesse_Moyenne", "Pluie", "Circuit", "Nuit", "Virages", "Longueur_km", "Circuit_Lent", "Circuit_Moderé", "Circuit_Rapide", 
                        "position_gp_saison_precedente","chrono_gp_saison_precedente"]
    assert len(df.columns) == 22, "Le DataFrame doit contenir 22 colonnes"
    assert isinstance(df, pl.DataFrame), "Le résultat doit être un DataFrame Polars"
    assert df["Position"].max() <= 24, "La position maximale devrait être inférieure ou égale à 24 (maximum 12 écuries de 2 pilotes)"


def test_plantage_nettoyage_dataset():
    with pytest.raises(FileNotFoundError):
        nettoyage_dataset(data_set_circuit="dataset\circuits.json", data_set_qualifs="data_set_qualifs.json")