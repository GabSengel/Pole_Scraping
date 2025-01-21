from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from typing import List, Set, Dict
from dataclasses import dataclass
from bs4 import BeautifulSoup

import time
import random
import json
import re

@dataclass
class InfoCircuit:
    """
    Représente les informations d'un circuit de Formule 1.

    Attributs :
        nom (str) : Le nom du circuit.
        lien (str) : Le lien vers la page détaillée du circuit.
        nombre_virages (str) : Le nombre de virages sur le circuit.
        longueur (str) : La longueur du circuit (exprimée en km ou m).

    Raises :
        ValueError : Si un des champs n'est pas une chaîne de caractères.
        ValueError : Si un des champs est vide ou contient uniquement des espaces.

    Exemple :
        >>> circuit = InfoCircuit(
        ...     nom="Monaco",
        ...     lien="https://fr.wikipedia.org/wiki/Circuit_de_Monaco",
        ...     nombre_virages="19",
        ...     longueur="3,337km"
        ... )
    """
    nom: str
    lien: str
    nombre_virages: str
    longueur: str

    def __post_init__(self):
        if not all(isinstance(attr, str) for attr in [self.nom, self.lien, self.nombre_virages, self.longueur]):
            raise ValueError("Tous les champs doivent être des chaînes de caractères.")

        if not all(attr.strip() for attr in [self.nom, self.lien, self.nombre_virages, self.longueur]):
            raise ValueError("Aucun champ ne doit être vide.")

    def to_dict(self):
        return {
            "Nom Circuit": self.nom,
            "Lien Circuit": self.lien,
            "Nombre de Virages": self.nombre_virages,
            "Longueur": self.longueur
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


def Recuperation_liens_circuit_wiki(lien: str) -> list:
    """
    Récupère les liens des circuits de Formule 1 à partir d'une page Wikipédia.

    Args :
        lien (str) : URL de la page Wikipédia contenant les informations sur les circuits.

    Returns :
        list : Une liste de dictionnaires contenant les noms et les liens des circuits.
               Chaque dictionnaire a la structure suivante :
               {
                   "Nom Circuit": str,
                   "Lien circuit": str
               }

    Exemple :
        >>> lien = "https://fr.wikipedia.org/wiki/Liste_alphab%C3%A9tique_des_circuits_de_Formule_1"
        >>> circuits = Recuperation_liens_circuit_wiki(lien)
        >>> print(circuits)
        [
            {"Nom Circuit": "Monaco", "Lien circuit": "https://fr.wikipedia.org/wiki/Circuit_de_Monaco"},
            {"Nom Circuit": "Spa-Francorchamps", "Lien circuit": "https://fr.wikipedia.org/wiki/Circuit_de_Spa-Francorchamps"}
        ]

    Note :
        - La fonction ignore les 4 premiers liens trouvés dans la page (considérés comme non pertinents).
        - Si la div contenant les liens (`navbox-container`) est absente, une liste vide est retournée.
        - La fonction utilise Selenium pour charger la page et BeautifulSoup pour l'analyse HTML.

    """
    options = webdriver.ChromeOptions()
    options.page_load_strategy = 'normal'
    driver = webdriver.Chrome(options=options)
    driver.maximize_window()

    wait_random()

    driver.get(lien)

    soup = BeautifulSoup(driver.page_source, "html.parser")

    navbox_container = soup.find("div", class_="navbox-container")
    if not navbox_container:
        driver.quit()
        return []

    elements_tableau = navbox_container.find_all("a")

    prefixe = "https://fr.wikipedia.org/"

    liste_id_circuit = list()

    for lien in elements_tableau[4:]:
        nom_circuit = lien.get_text(strip=True)
        href = lien.get("href")
        if not href: 
            continue
        lien_circuit = prefixe + href
        id_circuit = {
            "Nom Circuit": nom_circuit,
            "Lien circuit": lien_circuit
        }
        liste_id_circuit.append(id_circuit)

    driver.quit()

    return liste_id_circuit


def Recuperation_virage_longueur_circuit_wiki(liste_id_circuit: list) -> list:
    """
    Récupère les informations détaillées (nombre de virages et longueur) pour une liste de circuits à partir de leurs pages Wikipédia.

    Args :
        liste_id_circuit (list) : Une liste de dictionnaires contenant les noms et les liens des circuits.
            Chaque dictionnaire doit avoir la structure suivante :
            {
                "Nom Circuit": str,
                "Lien circuit": str
            }

    Returns :
        list : Une liste d'instances de la classe `InfoCircuit` contenant les informations détaillées sur chaque circuit :
            - `nom` : Nom du circuit.
            - `lien` : Lien vers la page du circuit.
            - `nombre_virages` : Nombre de virages (ou "Non trouvé" si absent).
            - `longueur` : Longueur du circuit (ou "Non trouvée" si absente).

    Exemple :
        >>> liste_id_circuit = [
        ...     {"Nom Circuit": "Monaco", "Lien circuit": "https://fr.wikipedia.org/wiki/Circuit_de_Monaco"},
        ...     {"Nom Circuit": "Spa-Francorchamps", "Lien circuit": "https://fr.wikipedia.org/wiki/Circuit_de_Spa-Francorchamps"}
        ... ]
        >>> circuits = Recuperation_virage_longueur_circuit_wiki(liste_id_circuit)
        >>> print(circuits)
        [
            InfoCircuit(nom="Monaco", lien="https://fr.wikipedia.org/wiki/Circuit_de_Monaco", nombre_virages="19", longueur="3,337km"),
            InfoCircuit(nom="Spa-Francorchamps", lien="https://fr.wikipedia.org/wiki/Circuit_de_Spa-Francorchamps", nombre_virages="20", longueur="7,004km")
        ]

    Notes :
        - La fonction utilise Selenium pour charger les pages des circuits et BeautifulSoup pour extraire les données HTML.
        - Si le nombre de virages ou la longueur est absent, "Non trouvé" ou "Non trouvée" est attribué par défaut.
        - Les pauses aléatoires avec `wait_random` sont insérées pour éviter une détection de scraping par Wikipédia.

    Raises :
        ValueError : Si les données extraites ne respectent pas les contraintes de la classe `InfoCircuit`.

    """
    options = webdriver.ChromeOptions()
    options.page_load_strategy = 'normal'
    driver = webdriver.Chrome(options=options)
    driver.maximize_window()

    wait_random()

    circuits = []

    for circuit in liste_id_circuit:

        lien = circuit["Lien circuit"]

        driver.get(lien)
        
        wait_random()

        soup = BeautifulSoup(driver.page_source, "html.parser")

        balise_th_virage = soup.find("th", string=re.compile("Nombre de virages", re.IGNORECASE))
        if balise_th_virage:
            nbr_virages = balise_th_virage.find_next("td").get_text(strip=True)
        else:
            nbr_virages = "Non trouvé"

        balise_th_longueur = soup.find("th", string=re.compile("Longueur", re.IGNORECASE))
        if balise_th_longueur:
            longueur = balise_th_longueur.find_next("td").get_text(strip=True)
        else:
            longueur = "Non trouvée"

        circuit_data = InfoCircuit(
            nom=circuit["Nom Circuit"],
            lien=circuit["Lien circuit"],
            nombre_virages=nbr_virages,
            longueur=longueur
        )
        circuits.append(circuit_data)

        wait_random()

    driver.quit()

    return circuits

if __name__ == "__main__":

    id_circuit = Recuperation_liens_circuit_wiki("https://fr.wikipedia.org/wiki/Liste_alphab%C3%A9tique_des_circuits_de_Formule_1")
    info_circuit = Recuperation_virage_longueur_circuit_wiki(id_circuit)

    info_circuit_dicts = [circuit.to_dict() for circuit in info_circuit]

    with open("data/results_scrap/data_set_infos_circuits3.json", "w", encoding="utf-8") as file:
        json.dump(info_circuit_dicts, file, ensure_ascii=False, indent=4)
