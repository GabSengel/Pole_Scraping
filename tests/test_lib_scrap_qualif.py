"""WARNING : Cette session de test lance selenium webdriver, plus précisement le driver Chrome !"""
import pytest
from src.scrapping.lib_scrap_qualif import (
    QualificationResult,
    Resultat_Qualif,
    Recherche_liens_GP_F1_stats,
    Resultats_Globaux
)

def test_init_QualificationResult():
    resultat = QualificationResult(
        Pilote = "Max VERSTAPPEN",
        GP = "Bahreïn",
        Saison = "2024",
        Châssis = "Red Bull",
        Moteur = "Honda RBPT",
        Pos = "1",
        Temps = "1'29.179",
        Écart = "",
        Moyenne = "218.473",
        Pluie = "Temps minimum pour être qualifié: 1'28.171 (107 % du meilleur temps en Q1: 1'34.342)G. Zhou n'a pas réalisé de temps en Q1 suite à un accident lors des essais libres.N. Hülkenberg n'a pas réalisé de temps en Q2 suite à un problème mécanique.",
        Circuit = "Sakhir",
        Date_gp = "samedi 2 mars 2024",
        Nuit = "Nuit"
)
    assert isinstance(resultat,QualificationResult)

def test_plantage_QualificationResult():
    with pytest.raises(ValueError):
        QualificationResult(
        Pilote = None,
        GP = "Bahreïn",
        Saison = "2024",
        Châssis = "Red Bull",
        Moteur = "Honda RBPT",
        Pos = "1",
        Temps = "1'29.179",
        Écart = "",
        Moyenne = "218.473",
        Pluie = "Temps minimum pour être qualifié: 1'28.171 (107 % du meilleur temps en Q1: 1'34.342)G. Zhou n'a pas réalisé de temps en Q1 suite à un accident lors des essais libres.N. Hülkenberg n'a pas réalisé de temps en Q2 suite à un problème mécanique.",
        Circuit = "Sakhir",
        Date_gp = "samedi 2 mars 2024",
        Nuit = "Nuit"
        )

    with pytest.raises(ValueError):
        QualificationResult(
        Pilote = "Max VERSTAPPEN",
        GP = "Bahreïn",
        Saison = "2024",
        Châssis = "Red Bull",
        Moteur = "Honda RBPT",
        Pos = 1,
        Temps = "1'29.179",
        Écart = "",
        Moyenne = 218.473,
        Pluie = "Temps minimum pour être qualifié: 1'28.171 (107 % du meilleur temps en Q1: 1'34.342)G. Zhou n'a pas réalisé de temps en Q1 suite à un accident lors des essais libres.N. Hülkenberg n'a pas réalisé de temps en Q2 suite à un problème mécanique.",
        Circuit = "Sakhir",
        Date_gp = "samedi 2 mars 2024",
        Nuit = "Nuit"
        )

    with pytest.raises(TypeError):
        QualificationResult(
        Pilote = "Max VERSTAPPEN",
        GP = "Bahreïn"
        )

def test_recherche_liens_GP_F1_stats():
    url_test = "https://www.statsf1.com/fr/default.aspx"
    result = Recherche_liens_GP_F1_stats(saison_depart="2024", nb_saisons=1, lien=url_test)
    assert len(result) > 0, "Le test a échoué : Aucun lien trouvé."

def test_plantage_recherche_liens_GP_F1_stats():
    url_invalide = "https://www.statsf1.com/invalide.aspx"
    result = Recherche_liens_GP_F1_stats(saison_depart="2024", nb_saisons=1, lien=url_invalide)
    assert result == [], "Le test a échoué : La fonction ne doit pas trouver de liens avec une URL invalide."

def test_resultat_qualif():
    page_source = """
    <html>
        <body>
            <h2>Vide</h2>
            <h2>Grand Prix de Monaco 2023</h2>
            <table>
                <tr>
                    <th>Pos</th><th>Pilote</th><th>Châssis</th><th>Moteur</th><th>Temps</th><th>Écart</th><th>Moyenne</th>
                </tr>
                <tr>
                    <td>1</td><td>Max Verstappen</td><td>Red Bull Racing</td><td>Honda</td>
                    <td>1:10.342</td><td>0.045</td><td>213.5 km/h</td>
                </tr>
            </table>
        </body>
    </html>
    """
    result = Resultat_Qualif(page_source)

    expected = [
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
            Pluie="vide",
            Circuit="",
            Date_gp="",
            Nuit=""
        )
    ]
    assert result == expected, f"Cas 1 échoué : Résultat attendu {expected}, obtenu {result}"

    page_source_incomplete = """
    <html>
        <body>
            <h2>Vide</h2>
            <h2>Grand Prix de Monaco 2023</h2>
            <table>
                <tr>
                    <th>Pos</th><th>Pilote</th><th>Châssis</th><th>Moteur</th><th>Temps</th><th>Écart</th><th>Moyenne</th>
                </tr>
                <tr>
                    <td>1</td><td>Max Verstappen</td><td></td><td>Honda</td>
                    <td></td><td></td><td></td>
                </tr>
            </table>
        </body>
    </html>
    """
    result_incomplete = Resultat_Qualif(page_source_incomplete)

    expected_incomplete = [
        QualificationResult(
            Pilote="Max Verstappen",
            GP="Grand Prix de Monaco",
            Saison="2023",
            Châssis="",
            Moteur="Honda",
            Pos="1",
            Temps="",
            Écart="",
            Moyenne="",
            Pluie="vide",
            Circuit="",
            Date_gp="",
            Nuit=""
        )
    ]
    assert result_incomplete == expected_incomplete, (
        f"Cas 2 échoué : Résultat attendu {expected_incomplete}, obtenu {result_incomplete}"
    )

def test_plantage_resultat_qualif():
    page_source_empty = "" 
    result = Resultat_Qualif(page_source_empty)
    assert result == [], "La fonction devrait retourner une liste vide pour une page HTML vide."

    page_source_no_table = """
    <html>
        <body>
            <h2>Vide</h2>
            <h2>Grand Prix de Monaco 2023</h2>
            <p>Pas de tableau disponible.</p>
        </body>
    </html>
    """
    result = Resultat_Qualif(page_source_no_table)
    assert result == [], "La fonction devrait retourner une liste vide si aucun tableau n'est trouvé."

    page_source_malformed_rows = """
    <html>
        <body>
            <h2>Vide</h2>
            <h2>Grand Prix de Monaco 2023</h2>
            <table>
                <tr>
                    <th>Pos</th><th>Pilote</th><th>Châssis</th><th>Moteur</th><th>Temps</th><th>Écart</th><th>Moyenne</th>
                </tr>
                <tr>
                    <td>1</td><td>Max Verstappen</td><td>Red Bull Racing</td><td>Honda</td>
                    <td>1:10.342</td><td>0.045</td><td>213.5 km/h</td>
                </tr>
                <tr>
                    <td>2</td><td>Charles Leclerc</td>  <!-- Ligne mal formée -->
                </tr>
            </table>
        </body>
    </html>
    """
    result = Resultat_Qualif(page_source_malformed_rows)
    expected_malformed = [
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
            Pluie="vide",
            Circuit="",
            Date_gp="",
            Nuit=""
        )
    ]
    assert result == expected_malformed, "La fonction devrait ignorer les lignes mal formées."

    page_source_no_weather = """
    <html>
        <body>
            <h2>Vide</h2>
            <h2>Grand Prix de Monaco 2023</h2>
            <table>
                <tr>
                    <th>Pos</th><th>Pilote</th><th>Châssis</th><th>Moteur</th><th>Temps</th><th>Écart</th><th>Moyenne</th>
                </tr>
                <tr>
                    <td>1</td><td>Max Verstappen</td><td>Red Bull Racing</td><td>Honda</td>
                    <td>1:10.342</td><td>0.045</td><td>213.5 km/h</td>
                </tr>
            </table>
        </body>
    </html>
    """
    result = Resultat_Qualif(page_source_no_weather)
    expected_no_weather = [
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
            Pluie="vide",
            Circuit="",
            Date_gp="",
            Nuit=""
        )
    ]
    assert result == expected_no_weather, "La fonction devrait attribuer 'Pas d'information sur la pluie' si la météo est absente."

def test_resultats_globaux():
    liens_test = [
        "https://www.statsf1.com/fr/2024/bahrein.aspx",
        "https://www.statsf1.com/fr/2024/arabie-saoudite.aspx"
    ]
    result = Resultats_Globaux(liens_test)

    assert len(result) > 0, "Le test a échoué : Aucun résultat collecté."
    assert all(isinstance(r, QualificationResult) for r in result), (
        "Le test a échoué : Tous les résultats doivent être des instances de QualificationResult."
    )
    example = result[0]
    assert example.Circuit != "", "Le test a échoué : Le champ 'Circuit' ne doit pas être vide."
    assert example.Date_gp != "", "Le test a échoué : Le champ 'Date_gp' ne doit pas être vide."
    assert example.Nuit != "", "Le test a échoué : Le champ 'Nuit' ne doit pas être vide."

def test_plantage_resultats_globaux():
    liens_test = []
    result = Resultats_Globaux(liens_test)
    assert result == [], "Le test a échoué : La fonction doit retourner une liste vide si aucun lien n'est fourni."
