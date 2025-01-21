import streamlit as st
import plotly.express as px
import polars as pl
from src.app.fonction_streamlit_app import (
    load_data,
    load_data_metrics,
    load_data_resultats_modeles,
    load_modele,
    load_data_param_modeles,
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

def main():
    st.set_page_config(page_title="Pôle Scraping", layout="wide", page_icon= "🚥")

    def page_acceuil():
        st.title("Pole Scraping 🏎️")
        st.markdown("""
        Bienvenue sur **Pole Scraping** ! Cette application vous permet de :
        - Explorer et visualiser les données de qualification des Grands Prix de Formule 1 des 10 dernières années.
        - Prédire la grille du prochain Grand Prix en fonction des données historiques et des conditions de course.

        **Vous êtes au bon endroit pour analyser les performances et anticiper les résultats !**
        """)

        col1, col2 = st.columns([1, 1], gap="small")

        with col1:
            st.subheader("Data Visualization 📊")
            st.write("""
            Dans cette section, vous pouvez explorer et visualiser les données de qualification
            des 10 dernières années. Découvrez nos données récoltées ainsi que l'historique des performances des pilotes
            et des écuries sur leurs derniers GP.
            """)
            if st.button("Data Visualization"):
                st.session_state["page"] = "Data Viz"
                st.rerun()

        with col2:
            st.subheader("Prédiction d'une grille de GP 🔮")
            st.write("""
            Ici, vous pourrez prédire la grille de départ pour un Grand Prix futur.
            Entrez les conditions de course et les caractéristiques des pilotes pour 
            obtenir une estimation des positions et des chronos.
            """)
            if st.button("Prédiction"):
                st.session_state["page"] = "Prediction"
                st.rerun()

    def page_data_viz():
        st.title("Data Visualization 📊")
        st.markdown("""
        Explorez différents graphiques liés à l'historique des résultats en qualification.
        """)
        df = load_data()

        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
            [   
                "📋 Données",
                "🥇 Historique des positions",
                "🌍 Historique des positions par GP",
                "📆 Position moyenne par Saison",
                "⚔️ Pilote vs Pilote",
                "🐎 Ecurie vs Ecurie",
                "🔗 Matrice de corélation"
            ]
        )

        with tab1 :
            df2 = df.with_columns(pl.col("Saison").cast(pl.Utf8))
            st.markdown(f"Nous avons récolté des informations sur les qualifications des {df.select(pl.col('Saison').n_unique()).item()} dernière années saison, soit {len(df)} résultats")
            st.dataframe(df2,height=550)

        with tab2:
            st.header("Historique des positions")

            default_pilote = "Lewis HAMILTON"
            pilote = st.selectbox("Sélectionnez un pilote :", sorted(df.select(pl.col("Pilote")).unique().to_series().to_list()),
                                index=sorted(df.select(pl.col("Pilote")).unique().to_series().to_list()).index(default_pilote))
            nombres_gp = st.slider("Nombre de GP à afficher :", 1, len(df.filter(pl.col("Pilote")==pilote).select(pl.col("GP")).to_series().to_list()), 10)

            fig = historic_positions(pilote, nombres_gp, df)
            st.plotly_chart(fig, use_container_width=True)

            moyenne = position_moyenne(Pilote=pilote,nombres_gp=nombres_gp, df=df)
            st.metric(f"Position moyenne de {pilote} sur ses {nombres_gp} derniers GP", value = moyenne)

        with tab3:
            st.header("Historique des positions par GP")

            col1, col2 = st.columns([1, 1], gap="small")

            default_pilote = "Charles LECLERC"
            default_gp = "Monaco"
            with col1 :
                pilote = st.selectbox("Sélectionnez un pilote :", sorted(df.select(pl.col("Pilote")).unique().to_series().to_list()),
                                index=sorted(df.select(pl.col("Pilote")).unique().to_series().to_list()).index(default_pilote))
            with col2 :
                grand_prix = st.selectbox("Sélectionner un GP", sorted(df.select(pl.col("GP")).unique().to_series().to_list()),
                                index=sorted(df.select(pl.col("GP")).unique().to_series().to_list()).index(default_gp))

            fig = historic_positions_par_gp(Pilote=pilote, GP=grand_prix, df=df)
            st.plotly_chart(fig, use_container_width=True)

            moyenne = position_moyenne_par_GP(Pilote=pilote, GP= grand_prix, df=df)
            st.metric(f"Position moyenne de {pilote} au GP de {grand_prix}", value=moyenne)
        
        with tab4:
            st.header("Position moyenne par Saison")

            default_pilote = "Max VERSTAPPEN"

            pilote = st.selectbox("Sélectionnez un pilote :", sorted(df.select(pl.col("Pilote")).unique().to_series().to_list()),
                        index=sorted(df.select(pl.col("Pilote")).unique().to_series().to_list()).index(default_pilote))
            fig = position_moyenne_par_saison(Pilote=pilote, df=df)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab5 : 
            st.header("Comparaison Pilote vs Pilote")
            default_saison = 2024
            default_pilote1 = "Max VERSTAPPEN"
            default_pilote2 = "Lewis HAMILTON"
            saison = st.selectbox("Sélectionnez une Saison :", sorted(df.select(pl.col("Saison")).unique().to_series().to_list()),
                        index=sorted(df.select(pl.col("Saison")).unique().to_series().to_list()).index(default_saison),
                        key="saison_pilote_vs_pilote")
            
            col1, col2 = st.columns(2)
            with col1:
                pilote1 = st.selectbox("Sélectionnez le premier pilote :", sorted(df.select(pl.col("Pilote")).unique().to_series().to_list()),
                        index=sorted(df.select(pl.col("Pilote")).unique().to_series().to_list()).index(default_pilote1))
            with col2:
                pilote2 = st.selectbox("Sélectionnez le second pilote :", sorted(df.select(pl.col("Pilote")).unique().to_series().to_list()),
                        index=sorted(df.select(pl.col("Pilote")).unique().to_series().to_list()).index(default_pilote2))

            stats_p1 = stats_pilotes_saisons(pilote1, saison,df)
            stats_p2 = stats_pilotes_saisons(pilote2, saison,df)

            st.subheader(f"Statistiques : {pilote1} vs {pilote2} pour la saison {saison}")

            col3, col4 = st.columns(2)

            with col3:
                st.metric("Position Moyenne", stats_p1[0])
                st.metric("Position Moyenne (Pluie)", stats_p1[1])
                st.metric("Vitesse Moyenne (Circuit Lent)", stats_p1[2])
                st.metric("Vitesse Moyenne (Circuit Modéré)", stats_p1[3])
                st.metric("Vitesse Moyenne (Circuit Rapide)", stats_p1[4])

            with col4:
                st.metric("Position Moyenne", stats_p2[0])
                st.metric("Position Moyenne (Pluie)", stats_p2[1])
                st.metric("Vitesse Moyenne (Circuit Lent)", stats_p2[2])
                st.metric("Vitesse Moyenne (Circuit Modéré)", stats_p2[3])
                st.metric("Vitesse Moyenne (Circuit Rapide)", stats_p2[4])

        with tab6 : 
            st.header("Comparaison Ecurie vs Ecurie")
            default_saison = 2024
            default_ecurie1 = "Red Bull"
            default_ecurie2 = "McLaren"
            saison = st.selectbox("Sélectionnez une Saison :", sorted(df.select(pl.col("Saison")).unique().to_series().to_list()),
                        index=sorted(df.select(pl.col("Saison")).unique().to_series().to_list()).index(default_saison),
                        key="saison_ecurie_vs_ecurie")
            
            col1, col2 = st.columns(2)
            with col1:
                ecurie1 = st.selectbox("Sélectionnez la première écurie :", sorted(df.select(pl.col("Chassis")).unique().to_series().to_list()),
                        index=sorted(df.select(pl.col("Chassis")).unique().to_series().to_list()).index(default_ecurie1))
            with col2:
                ecurie2 = st.selectbox("Sélectionnez la seconde écurie :", sorted(df.select(pl.col("Chassis")).unique().to_series().to_list()),
                        index=sorted(df.select(pl.col("Chassis")).unique().to_series().to_list()).index(default_ecurie2))

            stats_e1 = stats_ecuries_saisons(ecurie1, saison,df)
            stats_e2 = stats_ecuries_saisons(ecurie2, saison,df)

            st.subheader(f"Statistiques : {ecurie1} vs {ecurie2} pour la saison {saison}")

            col3, col4 = st.columns(2)

            with col3:
                st.metric("Position Moyenne", stats_e1[0])
                st.metric("Position Moyenne (Pluie)", stats_e1[1])
                st.metric("Vitesse Moyenne (Circuit Lent)", stats_e1[2])
                st.metric("Vitesse Moyenne (Circuit Modéré)", stats_e1[3])
                st.metric("Vitesse Moyenne (Circuit Rapide)", stats_e1[4])

            with col4:
                st.metric("Position Moyenne", stats_e2[0])
                st.metric("Position Moyenne (Pluie)", stats_e2[1])
                st.metric("Vitesse Moyenne (Circuit Lent)", stats_e2[2])
                st.metric("Vitesse Moyenne (Circuit Modéré)", stats_e2[3])
                st.metric("Vitesse Moyenne (Circuit Rapide)", stats_e2[4])
        
        with tab7 :
            fig = correlation_matrix(df)
            st.plotly_chart(fig, use_container_width=True)

    def page_prediction():
        st.title("Prédiction 🔮")
        st.markdown("""
        Cette section vous permet de prédire les résultats en qualification d'un futur Grand Prix
        """)
        tab1, tab2 = st.tabs(
            [   
                "🔍 Présentation de notre modèle",
                "🚥 Prédiction d'une Grille",
            ]
        )
        with tab1 :
            st.markdown("Cette partie a pour but de présenter en détails les caractéristiques du modèle de prédiction que nous avons selectionné et entrainé")
            st.markdown("""Si vous n'êtes pas un adepte de Machine Learning, je vous conseille de passer directement à la partie 'Prédiction d'une Grille'""")
            st.subheader(f'Résultat des prédictions de la variable "allure" de notre modèle Boosting sur une partie aléatoire des données')
            col1, inter_cols_space, col2 = st.columns((5,0.1,5))
            with col1:
                data = load_data_resultats_modeles("boosting_allure")
                data = data.with_columns(pl.col("Saison").cast(pl.Utf8))
                st.dataframe(data=data,height=400)
            with col2:
                st.dataframe(load_data_metrics("boosting_allure"), width=250)
                st.dataframe(load_data_param_modeles("boosting_allure"), width=250)
            with st.expander("Aide à la lecture"):
                st.subheader("1er tableau :")
                st.markdown("Le premier tableau permet de mettre face à face les valeurs réelles des valeurs prédites sur une partie des donénes qui a été sélectionné aléatoirement lors de l'entrainement du modèle. Exemple, lors du GP de Sao Paulo 2023, Nico Hulkenberg a été qualifié avec une allure de 16.37 secondes/km et le modèle a prédit une valeur de 16.69")
                st.subheader("2ème tableau :")
                st.markdown("RMSE (Root Mean Squared Error): Racine carrée de l'erreur quadratique moyenne. Mesure l'écart moyen entre les valeurs réelles et les valeurs prédites par le modèle. Plus il est faible, meilleure est la précision des prédictions.")
                st.markdown("**MAE (Mean Absolute Error): Erreur moyenne absolue entre les prédictions et les valeurs réelles. Contrairement au RMSE, il ne pénalise pas autant les grandes erreurs. Plus il est faible, plus les prédictions sont proches des valeurs réelles. Il est exprimée dans les mêmes unités que la cible.**")
                st.markdown("""<p style="color:red;">Il s'agit de la metric que nous avons voulu minimiser lors de l'entrainement et de la selection des hyperparamètres (plus d'informations dans "A Propos")</p>""", unsafe_allow_html=True)
                st.markdown("R² (Coefficient de détermination) : Le R² mesure la proportion de la variance des données expliquée par le modèle. Il donne une idée de la qualité globale de l’ajustement du modèle. Plus il est proche de 1, plus le modèle explique la variance des données")
                st.subheader("3ème tableau :")
                st.markdown("regressor__learning_rate (Taux d'apprentissage) : Contrôle la vitesse à laquelle le modèle ajuste les poids au cours de l'entraînement. Plus sa valeur est faible, plus l'entrainement sera lent mais il sera d'autant plus précis, réduisant le risque de surajustement. Plus la valeur est élevée, plus l'entraînement sera rapide mais cela risque d'augmenter le risque d'overfeeting.")
                st.markdown("regressor__max_depth (Profondeur maximale) : Limite la profondeur des arbres de décision dans le modèle Gradient Boosting. Une profondeur faible (ex. 3) permet de contrôler la complexité du modèle et de limiter l'overfitting. Une profondeur élevée permet au modèle de capturer des relations complexes, mais augmente le risque d'overfitting.")
                st.markdown("regressor__n_estimators (Nombre d'estimateurs) : Nombre total d'arbres dans le modèle Gradient Boosting. Un plus grand nombre d'arbres (ex. 500) peut améliorer la performance, mais allonge le temps d'entraînement. Un nombre trop faible d'arbres peut sous-ajuster les données.")
            nbr_de_variables = st.slider("Nombre de variables à afficher :", 1, 150, 10)
            st.plotly_chart(importance_variable(load_modele("boosting_allure"),top_n=nbr_de_variables))
        with tab2:
            df = load_data()

            nombre_pilotes = st.number_input(
                "Nombre de pilotes à inclure dans la prédiction",
                min_value=1,
                max_value=20,
                value=10,
                step=1
            )

            default_pilotes = [
                "Max VERSTAPPEN", "Lewis HAMILTON", "Charles LECLERC", "Fernando ALONSO",
                "Sergio PEREZ", "George RUSSELL", "Carlos SAINZ", "Lando NORRIS",
                "Esteban OCON", "Pierre GASLY"
            ]
            default_chassis = [
                "Red Bull", "Mercedes", "Ferrari", "Aston Martin",
                "Red Bull", "Mercedes", "Ferrari", "McLaren",
                "Renault/Alpine", "Renault/Alpine"
            ]
            default_moteurs = [
                "Honda", "Mercedes", "Ferrari", "Renault",
                "Honda", "Mercedes", "Ferrari", "Mercedes",
                "Renault", "Renault"
            ]

            default_pilotes = default_pilotes[:nombre_pilotes]
            default_chassis = default_chassis[:nombre_pilotes]
            default_moteurs = default_moteurs[:nombre_pilotes]

            st.subheader("Liste des pilotes")
            pilotes = []
            liste_pilotes = sorted(df.select(pl.col("Pilote")).unique().to_series().to_list())
            for i in range(nombre_pilotes):
                pilote = st.selectbox(
                    f"Pilote {i + 1}",
                    options=liste_pilotes,
                    index=liste_pilotes.index(default_pilotes[i]) if i < len(default_pilotes) else 0,
                    key=f"pilote_{i}"
                )
                pilotes.append(pilote)

            st.subheader("Grand Prix")
            liste_gps = sorted(df.select(pl.col("GP")).unique().to_series().to_list())
            gp = st.selectbox("Sélectionnez le Grand Prix :", options=liste_gps,
                            index=sorted(df.select(pl.col("GP")).unique().to_series().to_list()).index("Monaco"))

            saison = st.number_input("Saison", min_value=1950, max_value=2100, value=2025, step=1)
            numero_gp = st.number_input("Numéro du Grand Prix", min_value=1, value=6, step=1)

            st.subheader("Châssis et moteurs")
            chassis = []
            moteurs = []
            liste_chassis = sorted(df.select(pl.col("Chassis")).unique().to_series().to_list())
            for i, pilote in enumerate(pilotes):
                col1, col2 = st.columns(2)
                with col1:
                    chassis.append(st.selectbox(
                        f"Châssis de {pilote}",
                        options=liste_chassis,
                        index=liste_chassis.index(default_chassis[i]) if i < len(default_chassis) else 0,
                        key=f"chassis_{i}"
                    ))
                with col2:
                    moteurs.append(st.selectbox(
                        f"Moteur de {pilote}",
                        options=["Honda", "Mercedes", "Ferrari", "Renault"],
                        index=["Honda", "Mercedes", "Ferrari", "Renault"].index(default_moteurs[i]) if i < len(default_moteurs) else 0,
                        key=f"moteur_{i}"
                    ))

            st.subheader("Conditions de course")
            pluie = st.radio("Pluie", options=[0, 1], format_func=lambda x: "Oui" if x == 1 else "Non", index=0)
            nuit = st.radio("Course de nuit", options=[0, 1], format_func=lambda x: "Oui" if x == 1 else "Non", index=0)
            st.subheader("Circuit")
            liste_circuits = sorted(df.select(pl.col("Circuit")).unique().to_series().to_list())
            circuit = st.selectbox("Sélectionnez le circuit :", options=liste_circuits, index=liste_circuits.index("Monaco"))

            col1, col2 = st.columns(2)

            if st.button("Prédire les résultats", key="predict_button"):
                try:
                    prediction_df = preparation_data_prediction(
                        df = df.to_pandas() if isinstance(df, pl.DataFrame) else df,
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

                    model_path = "data/ml/models/model_boosting_allure.pkl"
                    prediction_results = prediction_grille(prediction_df, model_path)
                    sorted_predictions = ordonner_prediction(prediction_results, target="Allure")
                    import pandas as pd

                    sorted_predictions["Classement"] = range(1, len(sorted_predictions) + 1)

                    sorted_predictions = sorted_predictions[["Classement"] + [col for col in sorted_predictions.columns if col != "Classement"]]

                    st.subheader("Résultats des prédictions")
                    st.dataframe(sorted_predictions)

                except Exception as e:
                    st.error(f"Une erreur s'est produite lors de la prédiction : {e}")
    
    def page_a_propos() :
        st.title("""Informations sur le projet Pôle Scraping""")
        st.subheader("""Pourquoi Pôle Scraping ?""")
        st.markdown("""
        Dans le cadre de notre cours intitulé « Web Scraping et Machine Learning » de M2 MECEN à l'Université de Tours, nous avons choisi de réaliser un projet visant à prédire une grille de départ en Formule 1 (résultats des qualifications).  
        Cette idée nous est venue grâce à un jeu que nous avons créé au sein de notre groupe d'amis, tous fans de Formule 1. Ce jeu consiste à prédire la grille de départ du prochain Grand Prix en plaçant un maximum de pilotes à la bonne position. Déterminés à maximiser nos chances de victoire, nous avons décidé d'utiliser tous les outils à notre disposition pour améliorer nos prédictions. 
        """)
        st.markdown("""<p style="color:red;">(Avertissement : Ce projet n'a pas été réalisé dans le but de générer des gains dans les paris sportifs.)</p>""", unsafe_allow_html=True)
        st.subheader("""Pourquoi les qualifications ?""")
        st.markdown("""
        Nous avons opté pour les qualifications car elles sont confrontées à beaucoup moins d'aléas que la course. Les problèmes mécaniques y sont moins fréquents, et les accidents entre pilotes sont rares, car les qualifications ne reposent pas sur une confrontation directe entre eux.
        """)
        st.subheader("""Organisation du projet""")
        st.markdown("""
        **Le premier objectif** de ce projet a été de constituer une base de données. Pour cela, nous avons utilisé la méthode de web scraping, qui consiste à extraire des données directement depuis des sites internet afin de créer une base de données personnalisée. Cette dernière regroupe diverses informations sur les pilotes, les écuries et les circuits.
        """)
        st.markdown("""
        **La seconde partie** de ce projet repose sur l'utilisation du Machine Learning pour réaliser les prédictions. Bien que notre objectif soit de prédire les positions des pilotes, nous n'avons pas retenu la variable "position", car il s'agit d'une variable discrète et les modèles que nous avons essayé prédisent une variable continue. Nous avons donc opté pour la variable "temps_sec" (correspondant au meilleur temps au tour réalisé par un pilote lors des qualifications), à laquelle nous avons appliqué une modification.
        """)
        st.markdown("""
        Pour adapter cette variable, nous avons effectué une normalisation contextuelle en fonction de la longueur du circuit. Plus précisément, nous avons créé une nouvelle variable, "Allure", en divisant le temps en secondes par la longueur du circuit (en kilomètres) : temps_sec / longueur_km. Ainsi, notre modèle se concentre sur la prédiction de cette variable "Allure".
        """)
        st.markdown("""
        Une fois l'allure prédite pour chaque Pilote, il suffit de trier les résultats obtenus dans l'odre croissant de la variable allure et nous obtenons une grille de Grand Prix.
        """)
        st.markdown('<a id="plus-info"></a>', unsafe_allow_html=True)
        st.markdown("""
        La métrique cible utilisée dans notre modèle est la moyenne des erreurs absolues (Mean Absolute Error). Cette métrique a été choisie car elle attribue un poids égal à toutes les erreurs de prédiction, quelle que soit leur amplitude. Cela correspond parfaitement à notre contexte : dans notre jeu, se tromper d’une seule place ou de dix places entraîne le même résultat final. En minimisant cette métrique, notre modèle est conçu pour réduire globalement l’écart entre les prédictions et les valeurs réelles.
        """)
        st.subheader("""Utilisation de l'outil""")
        st.markdown("""Pour estimer une prédiction de grille, il vous suffit d'aller dans l'onglet "Prediction 🔮" puis dans la partie "🚥 Prédiction d'une Grille", de sélectionner le nombre de Pilotes présent, d'indiquer les informations les concernants (nom, écuries, moteur...) ainsi que les informations sur les conditions de courses et le circuit.  
                    Appuyer sur "Prédire les résultats" et l'application vous sortira sa prédiction.
        """)
        st.subheader("""Limite et idée d'amélioration""")
        st.markdown("""
        Actuellement, nos données ne regroupent des informations que sur les saisons précedentes car nous sommes dans l'off season. De ce fait l'estimation de la grille du 1er GP de la saison ne sera pas optimale car nous n'avons pas d'information sur les rookies qui vont faire leurs débuts cette années.  
        Pour améliorer la qualité des prédictions, nous avons comme objectif futur d'automatiser la mise à jour des données ainsi que l'entrainement de notre modèle dans le but de toujours avoir les données les plus récente possible pour qu'au cours d'une saison, nos prédiction deviennent de plus en plus proche de la réalité.""")
        st.subheader("👤 Auteurs")
        col1, col2, col3, col4 = st.columns(4)
        with col1 :
            st.markdown("""
            Gabin **SENGEL**  
            [Mon LinkedIn]({"https://www.linkedin.com/in/gabin-sengel/"})  
            [Mon Github]({"https://github.com/GabSengel"})  
            """)
        with col2 : 
            st.markdown("""
            Romain **SIETTE**  
            "[Mon LinkedIn]({"https://www.linkedin.com/in/romain-siette/"})
            """)
        st.text("MAJ le 2025/01/21")
    if "page" not in st.session_state:
        st.session_state.page = "Accueil"

    with st.sidebar:
        st.title("Navigation")

        accueil_type = "secondary" if st.session_state.page != "Accueil" else "primary"
        data_viz_type = "secondary" if st.session_state.page != "Data Viz" else "primary"
        prediction_type = "secondary" if st.session_state.page != "Prediction" else "primary"
        a_propos_type = "secondary" if st.session_state.page != "A Propos" else "primary"

        if st.button("Accueil 🏠", key="accueil", type=accueil_type):
            st.session_state.page = "Accueil"
            st.rerun()

        if st.button("Data Viz 📊", key="data_viz", type=data_viz_type):
            st.session_state.page = "Data Viz"
            st.rerun()

        if st.button("Prediction 🔮", key="prediction", type=prediction_type):
            st.session_state.page = "Prediction"
            st.rerun()

        if st.button("A Propos ℹ️ ", key="a_propos", type=a_propos_type):
            st.session_state.page = "A Propos"
            st.rerun()

    if st.session_state.page == "Accueil":
        page_acceuil()
    elif st.session_state.page == "Data Viz":
        page_data_viz()
    elif st.session_state.page == "Prediction":
        page_prediction()
    elif st.session_state.page == "A Propos" :
        page_a_propos()

if __name__ == "__main__":
    main()
