import pytest
from unittest.mock import patch, MagicMock
from src.scrapping.lib_scrap_circuit import (
    InfoCircuit,
    Recuperation_liens_circuit_wiki,
    Recuperation_virage_longueur_circuit_wiki)

def test_init_InfoCircuit(): 
    information_circuit = InfoCircuit(
        nom="Monaco", 
        lien="https://fr.wikipedia.org/wiki/Circuit_de_Monaco",
        nombre_virages="18",
        longueur="3,337km"
        )
    assert isinstance(information_circuit,InfoCircuit)

def test_plantage_InfoCircuit():
    with pytest.raises(ValueError):
        InfoCircuit(nom="Monaco", lien="https://fr.wikipedia.org/wiki/Circuit_de_Monaco", nombre_virages=18, longueur="3,337km")
    with pytest.raises(ValueError):
        InfoCircuit(nom="Monaco", lien="https://fr.wikipedia.org/wiki/Circuit_de_Monaco", nombre_virages="18", longueur=3)
    with pytest.raises(ValueError):
        InfoCircuit(nom=None, lien=None, nombre_virages=None,longueur=None)

@patch("src.scrapping.lib_scrap_circuit.webdriver.Chrome")
def test_Recuperation_liens_circuit_wiki_robuste(mock_chrome):
    mock_driver = MagicMock()
    mock_chrome.return_value = mock_driver
    mock_driver.page_source = """
    <div class="navbox-container">
        <ul class="liste-horizontale">
            <li><a href="/wiki/Circuit_international_de_Sakhir" title="Circuit international de Sakhir">Sakhir</a></li>
            <li><a href="/wiki/Circuit_international_de_Sakhir" title="Circuit international de Sakhir">Sakhir</a></li>
            <li><a href="/wiki/Circuit_international_de_Sakhir" title="Circuit international de Sakhir">Sakhir</a></li>
            <li><a href="/wiki/Circuit_international_de_Sakhir" title="Circuit international de Sakhir">Sakhir</a></li>
            <li><a href="/wiki/Circuit_international_de_Sakhir" title="Circuit international de Sakhir">Sakhir</a></li>
            <li><a href="/wiki/Circuit_de_la_corniche_de_Djeddah" title="Circuit de la corniche de Djeddah">Djeddah</a></li>
        </ul>
    </div>
    """
    result = Recuperation_liens_circuit_wiki("https://fake-url.com")
    expected = [
        {
            "Nom Circuit": "Sakhir",
            "Lien circuit": "https://fr.wikipedia.org//wiki/Circuit_international_de_Sakhir"
        },
        {
            "Nom Circuit": "Djeddah",
            "Lien circuit": "https://fr.wikipedia.org//wiki/Circuit_de_la_corniche_de_Djeddah"
        }
    ]
    assert result == expected

def test_plantage_Recuperation_liens_circuit_wiki():
    with patch("src.scrapping.lib_scrap_circuit.webdriver.Chrome") as mock_chrome:
        mock_driver = MagicMock()
        mock_chrome.return_value = mock_driver
        mock_driver.page_source = """
        <html>
            <body>
                <p>Pas de contenu pertinent ici</p>
            </body>
        </html>
        """
        result = Recuperation_liens_circuit_wiki("https://fake-url.com")
        assert result == [], "La div `navbox-container` est absente"
    
    with patch("src.scrapping.lib_scrap_circuit.webdriver.Chrome") as mock_chrome:
        mock_driver = MagicMock()
        mock_chrome.return_value = mock_driver
        mock_driver.page_source = """
        <div class="navbox-container">
            <ul class="liste-horizontale">
                <li>Pas de liens ici</li>
                <li>Pas de liens ici</li>
                <li>Pas de liens ici</li>
                <li>Pas de liens ici</li>
                <li>Pas de liens ici</li>
            </ul>
        </div>
        """
        result = Recuperation_liens_circuit_wiki("https://fake-url.com")
        assert result == [], "La div `navbox-container` ne contient aucun lien"

    with patch("src.scrapping.lib_scrap_circuit.webdriver.Chrome") as mock_chrome:
        mock_driver = MagicMock()
        mock_chrome.return_value = mock_driver
        mock_driver.page_source = """
        <div class="navbox-container">
            <ul class="liste-horizontale">
                <li><a title="Circuit sans href">Lien cassé</a></li>
                <li><a title="Circuit sans href">Lien cassé</a></li>
                <li><a title="Circuit sans href">Lien cassé</a></li>
                <li><a title="Circuit sans href">Lien cassé</a></li>
                <li><a title="Circuit sans href">Lien cassé</a></li>
                <li><a href="/wiki/Circuit_valide" title="Circuit valide">Circuit valide</a></li>
            </ul>
        </div>
        """
        result = Recuperation_liens_circuit_wiki("https://fake-url.com")
        expected = [
            {
                "Nom Circuit": "Circuit valide",
                "Lien circuit": "https://fr.wikipedia.org//wiki/Circuit_valide"
            }
        ]
        assert result == expected, "Un lien mal formé (sans href) n'a pas été ignoré correctement."
    
    with patch("src.scrapping.lib_scrap_circuit.webdriver.Chrome") as mock_chrome:
        mock_driver = MagicMock()
        mock_chrome.return_value = mock_driver
        mock_driver.page_source = ""
        result = Recuperation_liens_circuit_wiki("https://fake-url.com")
        assert result == [], "La page HTML est vide, mais le résultat n'est pas une liste vide."

    with patch("src.scrapping.lib_scrap_circuit.webdriver.Chrome") as mock_chrome:
        mock_driver = MagicMock()
        mock_chrome.return_value = mock_driver
        mock_driver.get.side_effect = Exception("Erreur Selenium")
        with pytest.raises(Exception, match="Erreur Selenium"):
            Recuperation_liens_circuit_wiki("https://fake-url.com")

def test_Recuperation_virage_longueur_circuit_wiki():
    with patch("src.scrapping.lib_scrap_circuit.webdriver.Chrome") as mock_chrome:
        mock_driver = MagicMock()
        mock_chrome.return_value = mock_driver
        mock_driver.page_source = """
        <html>
            <body>
                <table>
                    <tr><th>Nombre de virages</th><td>15</td></tr>
                    <tr><th>Longueur</th><td>5,412km</td></tr>
                </table>
            </body>
        </html>
        """

        liste_id_circuit = [
            {"Nom Circuit": "Sakhir", "Lien circuit": "https://fake-url.com/sakhir"},
            {"Nom Circuit": "Djeddah", "Lien circuit": "https://fake-url.com/djeddah"}
        ]

        result = Recuperation_virage_longueur_circuit_wiki(liste_id_circuit)
        expected = [
            InfoCircuit(nom="Sakhir", lien="https://fake-url.com/sakhir", nombre_virages="15", longueur="5,412km"),
            InfoCircuit(nom="Djeddah", lien="https://fake-url.com/djeddah", nombre_virages="15", longueur="5,412km")
        ]
        assert result == expected

    with patch("src.scrapping.lib_scrap_circuit.webdriver.Chrome") as mock_chrome:
        mock_driver = MagicMock()
        mock_chrome.return_value = mock_driver

        mock_driver.page_source = ""

        liste_id_circuit = [{"Nom Circuit": "Sakhir", "Lien circuit": "https://fake-url.com/sakhir"}]
        result = Recuperation_virage_longueur_circuit_wiki(liste_id_circuit)
        expected = [
            InfoCircuit(nom="Sakhir", lien="https://fake-url.com/sakhir", nombre_virages="Non trouvé", longueur="Non trouvée")
        ]
        assert result == expected, f"Cas 3 échoué : Résultat attendu {expected}, obtenu {result}"
    
    with patch("src.scrapping.lib_scrap_circuit.webdriver.Chrome") as mock_chrome:
        mock_driver = MagicMock()
        mock_chrome.return_value = mock_driver

        mock_driver.page_source = """
        <html>
            <body>
                <table>
                    <tr><th>Nombre de virages</th><td>15</td></tr>
                </table>
            </body>
        </html>
        """

        liste_id_circuit = [{"Nom Circuit": "Sakhir", "Lien circuit": "https://fake-url.com/sakhir"}]
        result = Recuperation_virage_longueur_circuit_wiki(liste_id_circuit)
        expected = [
            InfoCircuit(nom="Sakhir", lien="https://fake-url.com/sakhir", nombre_virages="15", longueur="Non trouvée")
        ]
        assert result == expected

def test_plantage_Recuperation_virage_longueur_circuit_wiki():
    with patch("src.scrapping.lib_scrap_circuit.webdriver.Chrome") as mock_chrome:
        mock_driver = MagicMock()
        mock_chrome.return_value = mock_driver
        mock_driver.get.side_effect = Exception("Erreur Selenium")
        
        liste_id_circuit = [{"Nom Circuit": "Sakhir", "Lien circuit": "https://fake-url.com/sakhir"}]
        with pytest.raises(Exception, match="Erreur Selenium"):
            Recuperation_virage_longueur_circuit_wiki(liste_id_circuit)
    
    with patch("src.scrapping.lib_scrap_circuit.webdriver.Chrome") as mock_chrome:
        mock_driver = MagicMock()
        mock_chrome.return_value = mock_driver
        mock_driver.page_source = """
        <html>
            <body>
                <table>
                    <tr><th>Autre Donnée</th><td>Valeur</td></tr>
                </table>
            </body>
        </html>
        """
        
        liste_id_circuit = [{"Nom Circuit": "Sakhir", "Lien circuit": "https://fake-url.com/sakhir"}]
        result = Recuperation_virage_longueur_circuit_wiki(liste_id_circuit)
        expected = [
            InfoCircuit(
                nom="Sakhir",
                lien="https://fake-url.com/sakhir",
                nombre_virages="Non trouvé",
                longueur="Non trouvée"
            )
        ]
        assert result == expected, "Cas 2 échoué : Les données manquantes ne sont pas correctement gérées."

    liste_id_circuit = []
    result = Recuperation_virage_longueur_circuit_wiki(liste_id_circuit)
    assert result == [], "Cas 3 échoué : Une liste vide en entrée devrait retourner une liste vide."

    with patch("src.scrapping.lib_scrap_circuit.webdriver.Chrome") as mock_chrome:
        mock_driver = MagicMock()
        mock_chrome.return_value = mock_driver
        mock_driver.page_source = """
        <html>
            <body>
                <table>
                    <tr><th>Nombre de virages</th><td>15</td></tr>
                    <tr><th>Longueur</th><td>5,412km</td></tr>
                </table>
            </body>
        </html>
        """
        
        liste_id_circuit = [
            {"Nom Circuit": "Sakhir"},
            {"Lien circuit": "https://fake-url.com/sakhir"}
        ]
        
        with pytest.raises(KeyError):
            Recuperation_virage_longueur_circuit_wiki(liste_id_circuit)

    with patch("src.scrapping.lib_scrap_circuit.webdriver.Chrome") as mock_chrome:
        mock_driver = MagicMock()
        mock_chrome.return_value = mock_driver
        mock_driver.page_source = """
        <html>
            <body>
                <table>
                    <tr><th>Nombre de virages</th><td>15</td></tr>
                    <tr><th>Longueur</th><td>5,412km</td></tr>
                </table>
            </body>
        </html>
        """
        with patch("src.scrapping.lib_scrap_circuit.BeautifulSoup") as mock_bs:
            mock_bs.side_effect = Exception("Erreur BeautifulSoup")
            
            liste_id_circuit = [{"Nom Circuit": "Sakhir", "Lien circuit": "https://fake-url.com/sakhir"}]
            with pytest.raises(Exception, match="Erreur BeautifulSoup"):
                Recuperation_virage_longueur_circuit_wiki(liste_id_circuit)

    with patch("src.scrapping.lib_scrap_circuit.webdriver.Chrome") as mock_chrome:
        mock_driver = MagicMock()
        mock_chrome.return_value = mock_driver
        mock_driver.page_source = """
        <html>
            <body>
                <table>
                    <tr><th>Nombre de virages</th><td>Non défini</td></tr>
                    <tr><th>Longueur</th><td>5,412km</td></tr>
                </table>
            </body>
        </html>
        """
        
        liste_id_circuit = [{"Nom Circuit": "Sakhir", "Lien circuit": "https://fake-url.com/sakhir"}]
        result = Recuperation_virage_longueur_circuit_wiki(liste_id_circuit)
        expected = [
            InfoCircuit(
                nom="Sakhir",
                lien="https://fake-url.com/sakhir",
                nombre_virages="Non défini",
                longueur="5,412km"
            )
        ]
        assert result == expected, "Cas 6 échoué : Les données incohérentes ne sont pas correctement gérées."





