Description de l'Application pour GitHub
Nom de l'Application : 📊 Dashboard Économétrique: Impact des Transferts de Fonds
Description :
Cette application est un dashboard interactif conçu pour analyser l'impact des transferts de fonds sur les indicateurs économiques et sociaux. Elle permet aux utilisateurs de charger des données, d'explorer les séries temporelles, de tester la stationnarité des variables, d'estimer des modèles économétriques (ARDL, VAR, VECM, Triple Moindres Carrés), de réaliser des simulations de stress tests et de générer des rapports complets. L'application est entièrement construite avec Streamlit, une bibliothèque Python pour créer des applications web interactives.
Fonctionnalités Principales
Importation des Données :
- Supporte les formats CSV, Excel, Stata (.dta) et SPSS (.sav).
- Interface intuitive pour sélectionner les variables et la variable temporelle.
Exploration des Données :
- Aperçu des données et statistiques descriptives.
- Visualisation des séries temporelles.
- Matrice de corrélation interactive.
Analyse de Stationnarité :
- Tests ADF et KPSS pour vérifier la stationnarité des séries.
- Différenciation automatique des séries non stationnaires.
Modélisation Économétrique :
- Estimation des modèles ARDL, VAR, VECM et Triple Moindres Carrés.
- Sélection automatique des lags basée sur l'AIC.
- Diagnostics des modèles (autocorrélation, hétéroscédasticité, normalité).
Simulations et Stress Testing :
- Simulation de chocs sur les variables sélectionnées.
- Fonctions de réponse impulsionnelle avec intervalles de confiance.
Génération de Rapports :
- Création de rapports PDF personnalisés.
- Inclut des graphiques, des statistiques descriptives et les résultats des modèles.
Interface Utilisateur :
- Design moderne et responsive.
- Onglets clairement organisés pour une navigation facile.
- Messages d'erreur et conseils de dépannage.
Technologies Utilisées
Streamlit : Pour l'interface utilisateur et l'interactivité.
Pandas : Pour la manipulation des données.
Statsmodels : Pour les modèles économétriques (ARDL, VAR, VECM).
Matplotlib et Seaborn : Pour les visualisations.
PyReadstat : Pour lire les fichiers Stata et SPSS.
FPDF : Pour générer des rapports PDF.
Scikit-learn : Pour la normalisation des données.
