import polars as pl 
import os

def nettoyage_dataset(data_set_circuit : str, data_set_qualifs : str) -> pl.DataFrame :
    # Import des dataset
    df_circuit = pl.read_json(data_set_circuit)
    df_qualifs = pl.read_json(data_set_qualifs)

    # Préparation pour la jointure des deux dataset
    df_circuit = df_circuit.rename({"Nom Circuit": "Circuit"})
    df_circuit = df_circuit.with_columns(
        pl.col("Circuit").str.replace("Mexico", "Mexico City")
    )
    df_qualifs = df_qualifs.with_columns(
        pl.col("Circuit").str.replace("Kuala Lumpur", "Sepang")
    ).with_columns(
        pl.col("Circuit").str.replace("Singapour", "Marina Bay")
    )

    # Jointure 
    df = df_qualifs.join(df_circuit, on="Circuit", how="left")
    df = df.drop("Lien circuit")

    # Nettoyage de la variable Pilote
    df = df.with_columns(
        pl.col("Pilote").str.replace(r"\*$", "")
    )
    df = df.filter(
        pl.col("Pilote") != "",
        pl.col("Pilote") != "Non qualifié"
    )

    # Nettoyage de la variable Moteur 
    df = df.with_columns(
        pl.col("Moteur").str.replace_all(r"RBPT|Honda RBPT", "Honda")
    ).with_columns(
        pl.col("Moteur").replace("BWT Mercedes", "Mercedes")
    ).with_columns(
        pl.col("Moteur").replace("TAG Heuer", "Renault")
    )

    # Nettoyage de la variable Châssis
    df = df.with_columns(
        pl.col("Châssis").str.replace_all(r"Sauber|Alfa Romeo|Kick Sauber", "Sauber")
    ).with_columns(
        pl.col("Châssis").str.replace(r"Force India|Racing Point|Aston Martin", "Aston Martin")
    ).with_columns(
        pl.col("Châssis").str.replace(r"Lotus|Renault|Alpine", "Renault/Alpine")
    ).with_columns(
        pl.col("Châssis").str.replace(r"Toro Rosso|AlphaTauri|RB", "Racing Bulls")
    )
    df = df.to_dummies(columns=["Moteur"])

    # Nettoyage de la variable Pos
    df = df.with_columns(
        pl.col("Pos").cast(pl.UInt8)
    )

    # Nettoyage de la variable Temps
    df = df.with_columns(
        pl.col("Temps").str.replace("-","0'00.000") 
    ).with_columns(
        pl.col("Temps").str.replace("'", ":")
    ).with_columns(
        (pl.col("Temps").str.extract(r"(\d+):(\d+\.\d+)", 1).cast(pl.Int32) * 60 +
        pl.col("Temps").str.extract(r"(\d+):(\d+\.\d+)", 2).cast(pl.Float64))
        .alias("Temps_sec")
    ).drop(
        "Temps" 
    ).select(
        [
            "Pilote", "GP", "Saison", "Châssis", "Moteur_Ferrari", "Moteur_Honda", "Moteur_Mercedes", "Moteur_Renault", "Pos", "Temps_sec",
            "Écart", "Moyenne", "Pluie", "Circuit", "Date_gp", "Nuit", "Nombre de virages", "Longueur"
        ]
    )

    for i in [0,0.01,0.02,0.03,0.04] :
        df = df.with_columns(
        pl.when(pl.col("Temps_sec") == i) #3
        .then(pl.col("Temps_sec").shift(1) + 0.010)
        .otherwise(pl.col("Temps_sec"))
        .alias("Temps_sec"))

    # Nettoyage de la variable Écart
    df = df.drop("Écart")

    # Nettoyage de la variable Longueur 
    df = df.with_columns(
        pl.col("Longueur").replace("de 0,826kmà 5,861km","5,861km")
    ).with_columns(
        pl.col("Longueur").str.replace("5,807km5,821km", "5,821km")
    ).with_columns(
        pl.col("Longueur").replace("5,513km(3,427mi)", "5,513km")
    ).with_columns(
        pl.col("Longueur").replace("5,848km[1]", "5,848km")
    ).with_columns(
        pl.col("Longueur").replace("(2021-Présent) 5,278km", "5,278km")
    ).with_columns(
        pl.col("Longueur").str.replace(r"km$", "")
    ).with_columns(
        pl.col("Longueur").str.replace(",", ".")
    )
    df = df.with_columns(
        pl.col("Longueur").cast(pl.Float64)
    )

    # Nettoyage de la variable Moyenne
    df = df.with_columns(
        pl.col("Moyenne").replace("","0")
    ).with_columns(
        pl.col("Moyenne").cast(pl.Float64)
    ).with_columns(
        pl.when(pl.col("Moyenne") == 0.0)
        .then((pl.col("Longueur") / pl.col("Temps_sec") * 3600).round(3))
        .otherwise(pl.col("Moyenne"))
        .alias("Moyenne")
    )
    vitesse_moyenne_par_circuit = df.group_by("Circuit").agg(
        pl.col("Moyenne")
        .mean()
        .round(4)
        .alias("Vitesse_Moyenne_Circuit")
    )
    df = df.join(vitesse_moyenne_par_circuit, on="Circuit", how="left")

    df = df.with_columns(
        pl.when(pl.col("Vitesse_Moyenne_Circuit") < 205)
        .then(1)
        .otherwise(0)
        .alias("Circuit_Lent"),
        pl.when((pl.col("Vitesse_Moyenne_Circuit") >= 205) & (pl.col("Vitesse_Moyenne_Circuit") <= 230))
        .then(1)
        .otherwise(0)
        .alias("Circuit_Moderé"),
        pl.when(pl.col("Vitesse_Moyenne_Circuit") > 230)
        .then(1)
        .otherwise(0)
        .alias("Circuit_Rapide")
    ).drop("Vitesse_Moyenne_Circuit")

    # Nettoyage de la variable Pluie
    df = df.with_columns(
        pl.when(pl.col("Pluie").str.contains("pluie"))
        .then(1) 
        .otherwise(0)
        .alias("Pluie")
    )

    # Nettoyage de la variable Nuit
    df = df.with_columns(
        pl.when(pl.col("Nuit").str.contains("Nuit"))
        .then(1) 
        .otherwise(0)
        .alias("Nuit")
    )

    # Nettoyage de la variable Nombre de virages
    df = df.with_columns(
        pl.col("Nombre de virages").replace("10 (anciennement 9 sans modification du tracé)[1],[2]","10")
    ).with_columns(
        pl.col("Nombre de virages").replace("18 (10 à droite et 8 à gauche)", "18")
    ).with_columns(
        pl.col("Nombre de virages").replace("16 (10 à droite et 6 à gauche)", "16")
    ).with_columns(
        pl.col("Nombre de virages").replace("19[1]", "19")
    ).with_columns(
        pl.col("Nombre de virages").replace("14\xa0Virages", "14")
    ).with_columns(
        pl.col("Nombre de virages").cast(pl.UInt8)
    )

    # Nettoyage de la variable Saison
    df = df.with_columns(
        pl.col("Saison").cast(pl.UInt32)
    )

    # Nettoyage de la variable Date_gp
    numeros_mois = {
        "janvier": "1", 
        "février": "2", 
        "mars": "3", 
        "avril": "4",
        "mai": "5", 
        "juin": "6", 
        "juillet": "7", 
        "août": "8",
        "septembre": "9", 
        "octobre": "10", 
        "novembre": "11", 
        "décembre": "12"
    }

    for mois, numero in numeros_mois.items():
        df = df.with_columns(
            pl.col("Date_gp").str.replace(mois, numero))

    df = df.with_columns(
    pl.col("Date_gp").str.replace_all(r"[a-zA-Z]", "").str.strip_chars()
    ).with_columns(
        pl.col("Date_gp").str.extract(r"^(\d+)", 0).alias("Jour"),
        pl.col("Date_gp").str.extract(r"\s(\d+)\s", 0).alias("Mois"),
        pl.col("Date_gp").str.extract(r"(\d+)$", 0).alias("Annee")    
    ).with_columns(
        pl.col("Jour").str.strip_chars().str.zfill(2).alias("Jour"),
        pl.col("Mois").str.strip_chars().str.zfill(2).alias("Mois"),
        pl.col("Annee").str.strip_chars().alias("Annee")
    ).with_columns(
        (pl.col("Annee") + "-" + pl.col("Mois") + "-" + pl.col("Jour"))
        .alias("Date_gp_standard")
    ).drop(
        ["Date_gp","Jour", "Mois", "Annee"]
    ).with_columns(
        pl.col("Date_gp_standard").cast(pl.Date)
    ).sort(
        ["Saison", "Date_gp_standard", "Pos"]
    ).with_columns(
        pl.col("Date_gp_standard")
        .rank("dense", descending=False)
        .over("Saison")  
        .cast(pl.UInt8)                   
        .alias("Numeros_gp")
    ).drop(
        "Date_gp_standard"
    )

    # Réorganisation des colonnes
    df = df.select(
        [
            "Pilote", "GP", "Saison", "Numeros_gp", "Châssis", "Moteur_Ferrari", "Moteur_Honda", "Moteur_Mercedes", "Moteur_Renault", "Pos", "Temps_sec",
            "Moyenne", "Pluie", "Circuit", "Nuit", "Nombre de virages", "Longueur", "Circuit_Lent", "Circuit_Moderé", "Circuit_Rapide"
        ]
    )
    df = df.rename({"Moyenne": "Vitesse_Moyenne"})
    df = df.rename({"Nombre de virages" : "Virages"})
    df = df.rename({"Châssis" : "Chassis"})
    df = df.rename({"Pos" : "Position"})
    df = df.rename({"Longueur" : "Longueur_km"})

    # Apporter des infos sur les GP précédents
    df = df.with_columns((df["Saison"] - 1).alias("Saison_Precedente"))

    df = df.join(
        df.select([
            "Pilote", 
            "GP", 
            "Saison", 
            "Position", 
            "Temps_sec"
        ]),
        how="left",
        left_on=["Pilote", "GP", "Saison_Precedente"],
        right_on=["Pilote", "GP", "Saison"],
        suffix="_prev"
    )

    df = df.rename({
        "Position_prev": "position_gp_saison_precedente",
        "Temps_sec_prev": "chrono_gp_saison_precedente"
    })

    df = df.drop("Saison_Precedente")

    min_saison = df["Saison"].min()

    df = df.filter(pl.col("Saison") > min_saison)

    df = df.with_columns([
    pl.col("position_gp_saison_precedente").fill_null(-1),
    pl.col("chrono_gp_saison_precedente").fill_null(-1.0),
    ])

    return df

if __name__ == "__main__":
    df_ml = nettoyage_dataset(data_set_circuit="data/results_scrap/data_set_infos_circuits.json", data_set_qualifs="data/results_scrap/data_set_qualifs2.json")
    
    output_path = os.path.join(os.path.dirname("data/clean_df/"), "DataFrame.json")

    df_ml.write_json(output_path)