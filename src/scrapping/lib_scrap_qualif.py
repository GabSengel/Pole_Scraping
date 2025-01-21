from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import List

import time
import random
import json
import re

@dataclass
class QualificationResult:
    """
    Représente les résultats de qualification d'un pilote lors d'un Grand Prix de Formule 1.

    Attributs :
        Pilote (str) : Nom du pilote.
        GP (str) : Nom du Grand Prix.
        Saison (str) : Année de la saison.
        Châssis (str) : Nom du châssis (i.e. l'écurie) du pilote.
        Moteur (str) : Nom du motoriste utilisé par le pilote.
        Pos (str) : Position obtenue lors des qualifications.
        Temps (str) : Temps réalisé par le pilote.
        Écart (str) : Écart de temps avec le meilleur pilote.
        Moyenne (str) : Vitesse moyenne réalisée lors des qualifications.
        Pluie (str) : Récupère des informations "extra" sur les qualifications (ex: conditions météorologiques, incidents, etc.).
        Circuit (str) : Nom du circuit.
        Date_gp (str) : Date du Grand Prix.
        Nuit (str) : Indique si l'événement a eu lieu de nuit.

    Méthodes :
        __post_init__ : Valide que tous les champs sont des chaînes de caractères.
        to_dict : Convertit l'objet en dictionnaire.

    Raises :
        ValueError : Si un des champs n'est pas une chaîne de caractères.

    Exemple :
        >>> resultat = QualificationResult(
        ...     Pilote="Max Verstappen",
        ...     GP="Grand Prix de Monaco",
        ...     Saison="2023",
        ...     Châssis="Red Bull Racing",
        ...     Moteur="Honda",
        ...     Pos="1",
        ...     Temps="1:10.342",
        ...     Écart="0.045",
        ...     Moyenne="213.5 km/h",
        ...     Pluie="Non",
        ...     Circuit="Monaco",
        ...     Date_gp="28 mai 2023",
        ...     Nuit="Non"
        ... )
        >>> print(resultat.to_dict())
        {
            "Pilote": "Max Verstappen",
            "GP": "Grand Prix de Monaco",
            "Saison": "2023",
            "Châssis": "Red Bull Racing",
            "Moteur": "Honda",
            "Pos": "1",
            "Temps": "1:10.342",
            "Écart": "0.045",
            "Moyenne": "213.5 km/h",
            "Pluie": "Non",
            "Circuit": "Monaco",
            "Date_gp": "28 mai 2023",
            "Nuit": "Non"
        }
    """
    Pilote: str
    GP: str
    Saison: str
    Châssis: str
    Moteur: str
    Pos: str
    Temps: str
    Écart: str
    Moyenne: str
    Pluie: str
    Circuit: str
    Date_gp: str
    Nuit: str

    def __post_init__(self):
        for field_name, value in self.__dict__.items():
            if not isinstance(value, str):
                raise ValueError(f"Le champ '{field_name}' doit être une chaîne de caractères.")
    
    def to_dict(self):
        return {
            "Pilote": self.Pilote,
            "GP": self.GP,
            "Saison": self.Saison,
            "Châssis": self.Châssis,
            "Moteur": self.Moteur,
            "Pos": self.Pos,
            "Temps": self.Temps,
            "Écart": self.Écart,
            "Moyenne": self.Moyenne,
            "Pluie": self.Pluie,
            "Circuit": self.Circuit,
            "Date_gp": self.Date_gp,
            "Nuit": self.Nuit
        }


def random_number() -> float:
    """
    Génère un nombre aléatoire flottant compris entre 1 et 2.

    Returns :
        float : Un nombre aléatoire entre 1 et 2.

    Exemple :
        >>> valeur = random_number()
        >>> 1 <= valeur <= 2
        True
    """
    number = random.uniform(1, 2)
    return number

def wait_random():
    """
    Met en pause l'exécution du programme Selenium pendant une durée aléatoire.

    La durée de la pause est comprise entre 1 et 2 secondes, générée par la fonction `random_number`.

    Exemple :
        >>> wait_random()  # Le programme s'arrête pour une durée aléatoire entre 1 et 2 secondes.
    """
    time.sleep(random_number())

def Recherche_liens_GP_F1_stats(saison_depart: str, nb_saisons: int, lien: str) -> list:
    """
    Récupère les liens des pages de Grands Prix de Formule 1 sur le site StatsF1 pour une ou plusieurs saisons.

    Args :
        saison_depart (str) : La saison de départ (exemple : "2024").
        nb_saisons (int) : Le nombre de saisons à analyser, y compris la saison de départ.
        lien (str) : L'URL de la page de départ sur StatsF1.

    Returns :
        list : Une liste de liens (str) correspondant aux pages des Grands Prix de F1.
               Exemple :
               [
                   "https://www.statsf1.com/fr/2024/gp1.aspx",
                   "https://www.statsf1.com/fr/2024/gp2.aspx"
               ]

    Exemple :
        >>> saison_depart = "2024"
        >>> nb_saisons = 2
        >>> lien = "https://www.statsf1.com/fr/default.aspx"
        >>> resultats = Recherche_liens_GP_F1_stats(saison_depart, nb_saisons, lien)
        >>> print(resultats)
        [
            "https://www.statsf1.com/fr/2024/gp1.aspx",
            "https://www.statsf1.com/fr/2024/gp2.aspx",
            "https://www.statsf1.com/fr/2023/gp1.aspx"
        ]

    Notes :
        - La fonction utilise Selenium pour naviguer sur le site StatsF1 et collecter les liens des Grands Prix.
        - Si un élément attendu est introuvable (bouton de navigation, saison, liens), la fonction retourne une liste vide.
        - Les pauses aléatoires avec `wait_random` sont utilisées pour éviter une détection de scraping par le site.

    Raises :
        NoSuchElementException : Si un élément attendu n'est pas trouvé (par exemple, bouton ou lien).
        Exception : Pour tout autre problème (par exemple, problème de connexion ou structure inattendue).

    """
    options = webdriver.ChromeOptions()
    options.page_load_strategy = 'normal'
    driver = webdriver.Chrome(options=options)
    driver.maximize_window()

    try:
        driver.get(lien)
        wait_random()

        try:
            boutton_saisons = driver.find_element(By.ID, "ctl00_HL_SeasonH")
            boutton_saisons.click()
        except NoSuchElementException:
            print("Bouton des saisons introuvable.")
            return []

        wait_random()
        liens_GP = []

        try:
            driver.find_element(By.LINK_TEXT, f"{saison_depart}").click()
        except NoSuchElementException:
            print("Saison introuvable.")
            return []

        wait_random()

        for _ in range(nb_saisons):
            try:
                grand_prix = driver.find_elements(By.XPATH, "//div[@class='flag']//a[contains(@href, '.aspx')]")
                for GP in grand_prix:
                    lien_GP = GP.get_attribute("href")
                    liens_GP.append(lien_GP)

                driver.find_element(By.ID, "ctl00_HL_NavigLeft").click()
            except NoSuchElementException:
                print("Navigation ou liens introuvables.")
                return []

            wait_random()

        return liens_GP

    except Exception as e:
        print(f"Erreur lors du chargement de la page : {e}")
        return []

    finally:
        driver.quit()

def Resultat_Qualif(page_source: str) -> List[QualificationResult]:
    """
    Extrait les résultats des qualifications d'un Grand Prix de Formule 1 à partir d'une page HTML.

    Args :
        page_source (str) : Le code source HTML de la page à analyser.

    Returns :
        List[QualificationResult] : Une liste d'instances de `QualificationResult` contenant les informations extraites pour chaque pilote.

    Exemple :
        >>> page_source = "<html>...</html>"  # Code HTML d'une page de qualification
        >>> resultats = Resultat_Qualif(page_source)
        >>> print(resultats)
        [
            QualificationResult(
                Pilote="Max Verstappen",
                GP="Grand Prix de Monaco",
                Saison="2023",
                Châssis="Red Bull Racing",
                Moteur="Honda",
                Pos="1",
                Temps="1:10.342",
                Écart="0.045",
                Moyenne="213.5 km/h",
                Pluie="Non",
                Circuit="",
                Date_gp="",
                Nuit=""
            )
        ]

    Notes :
        - La fonction utilise BeautifulSoup pour analyser la structure HTML.
        - Si certains champs (comme le GP ou la saison) ne sont pas trouvés, ils sont initialisés avec "vide".
        - Les données spécifiques aux pilotes sont extraites des balises `<tr>` et `<td>`.

    Exceptions gérées :
        - Si les balises attendues pour le Grand Prix ou la saison ne sont pas trouvées, la valeur "vide" est utilisée par défaut.
        - Si aucune donnée n'est trouvée dans la page HTML, une liste vide est retournée.

    """
    soup = BeautifulSoup(page_source, 'html.parser')
    try:
        course = soup.find_all("h2")[1].get_text()[:-5]
    except:
        course = "vide"
    
    try:
        saison = soup.find_all("h2")[1].get_text()[-4:]
    except:
        saison = "vide"

    pluie_element = soup.find(id="ctl00_CPH_Main_P_Commentaire")
    pluie = pluie_element.get_text(strip=True) if pluie_element else "vide"

    lignes = soup.find_all("tr")[1:]

    resultat_saison = []

    for ligne in lignes:
        colonnes = ligne.find_all("td")

        if len(colonnes) >= 6:
            donnees = QualificationResult(
                Pilote=colonnes[1].get_text(strip=True),
                GP=course,
                Saison=saison,
                Châssis=colonnes[2].get_text(strip=True),
                Moteur=colonnes[3].get_text(strip=True),
                Pos=colonnes[0].get_text(strip=True),
                Temps=colonnes[4].get_text(strip=True),
                Écart=colonnes[5].get_text(strip=True),
                Moyenne=colonnes[6].get_text(strip=True),
                Pluie=pluie,
                Circuit="",
                Date_gp="",
                Nuit=""
            )

            resultat_saison.append(donnees)

    return resultat_saison

def Resultats_Globaux(liens_GP: list) -> list:
    """
    Récupère les résultats globaux des qualifications de plusieurs Grands Prix de Formule 1.

    Args :
        liens_GP (list) : Une liste de liens (str) vers les pages des Grands Prix sur StatsF1.

    Returns :
        list : Une liste d'instances de `QualificationResult` contenant les informations détaillées pour chaque pilote
               dans les qualifications de tous les Grands Prix analysés.

    Exemple :
        >>> liens_GP = [
        ...     "https://www.statsf1.com/fr/2024/gp1.aspx",
        ...     "https://www.statsf1.com/fr/2024/gp2.aspx"
        ... ]
        >>> resultats = Resultats_Globaux(liens_GP)
        >>> print(resultats)
        [
            QualificationResult(
                Pilote="Max Verstappen",
                GP="Grand Prix de Monaco",
                Saison="2024",
                Châssis="Red Bull Racing",
                Moteur="Honda",
                Pos="1",
                Temps="1:10.342",
                Écart="0.045",
                Moyenne="213.5 km/h",
                Pluie="Non",
                Circuit="Monaco",
                Date_gp="28 mai 2024",
                Nuit="Oui"
            )
        ]

    Notes :
        - La fonction utilise Selenium pour naviguer sur les pages de qualification des Grands Prix et BeautifulSoup pour analyser leur contenu.
        - Les informations générales (nom du circuit, date et conditions nocturnes) sont extraites avant d'accéder aux données de qualification.
        - La fonction appelle `Resultat_Qualif` pour extraire les informations spécifiques aux pilotes.
        - Les pauses (`time.sleep`) sont utilisées pour éviter une détection de scraping par le site.

    Raises :
        Exception : Si un lien est inaccessible ou si la structure HTML diffère de ce qui est attendu.

    """
    options = webdriver.ChromeOptions()
    options.page_load_strategy = 'normal'
    driver = webdriver.Chrome(options=options)
    driver.maximize_window()

    if not liens_GP:
        print("Aucun lien fourni. Retour d'une liste vide.")
        return []
    
    resultat_globaux = []

    for lien in liens_GP:
        driver.get(lien)
        time.sleep(0.5)

        soup_intermédiaire = BeautifulSoup(driver.page_source, 'html.parser')
        circuit = [text for text in soup_intermédiaire.find("div", class_="GPinfo").stripped_strings][0]
        date_gp = [text for text in soup_intermédiaire.find("div", class_="GPinfo").stripped_strings][1]
        nuit = soup_intermédiaire.find("div", class_="GPmeteo").find_next("img").get("title")

        driver.find_element(By.ID, "ctl00_CPH_Main_HL_Qualification").click()
        time.sleep(0.5)

        resultat_qualif_GP = Resultat_Qualif(driver.page_source)

        for resultat in resultat_qualif_GP:
            resultat.Circuit = circuit
            resultat.Date_gp = date_gp
            resultat.Nuit = nuit

        resultat_globaux.extend(resultat_qualif_GP)

    driver.quit()

    return resultat_globaux

if __name__ == "__main__":
    lien = Recherche_liens_GP_F1_stats(saison_depart="2024", nb_saisons = 10, lien = "https://www.statsf1.com/fr/default.aspx")
    qualification = Resultats_Globaux(lien)

    qualification_dicts = [q.to_dict() for q in qualification]

    with open("data/results_scrap/data_set_qualifs2.json", "w", encoding="utf-8") as file:
        json.dump(qualification_dicts, file, ensure_ascii=False, indent=4)
