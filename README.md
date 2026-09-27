# Prediction de production photovoltaique (J+1)

![Build Status](https://github.com/eyabenmaiz/pv-production-prediction/actions/workflows/docker-build.yml/badge.svg)

Systeme intelligent de  prediction de la production photovoltaique a J+1, base sur un modele XGBoost entraine sur des donnees meteorologiques. Developpe initialement dans le cadre d'un Projet de Fin d'Annee (ENET'Com Sfax), puis conteneurise avec Docker et integre dans un pipeline CI/CD.

## Stack technique

- Machine Learning : XGBoost, scikit-learn, pandas, numpy
- Backend : Flask + Gunicorn (serveur de production)
- APIs externes : OpenWeatherMap (meteo), PVGIS (irradiance solaire)
- Conteneurisation : Docker
- CI/CD : GitHub Actions (build et publication automatique sur Docker Hub)

## Performances du modele

| Metrique | Valeur |
|---|---|
| R2 (test) | 0.698 |
| MAE | 0.538 kWh/kWp |
| RMSE | 0.783 |

## Lancer le projet

L'image Docker publiee sur Docker Hub contient deja le modele entraine : aucun besoin de cloner le depot ni de dataset.

    docker pull eyabenmaiz/pv-backend
    docker run -p 5000:5000 --env-file .env eyabenmaiz/pv-backend

Avant de lancer, cree un fichier .env a la racine de ton dossier de travail avec ta propre cle API OpenWeatherMap :

    OPENWEATHER_API_KEY=ta_cle_ici

## Utilisation de l'API

### Verifier l'etat du serveur

    GET /health

### Obtenir une prediction

    POST /predict
    Content-Type: application/json

    {
      "Capacite_kWp": 10,
      "Tavg": 25,
      "Tmax": 30,
      "Tmin": 20,
      "Wind_mps": 3,
      "Solar_Wm2": 6,
      "Humidity": 50,
      "Type_centrale": "Residentiel",
      "Region": "Tunisie/Sfax"
    }

Reponse :

    {
      "status": "success",
      "production_spec_kWh_per_kWp": 5.54,
      "production_totale_kWh": 55.4,
      "model_used": "XGBoost"
    }

## Code source

Le code source complet (backend Flask, script d'entrainement, Dockerfile, pipeline CI/CD) est disponible dans ce depot. Le dataset et les fichiers du modele entraine (model/*.pkl) ne sont pas versionnes (fichiers volumineux) mais sont inclus dans l'image Docker publiee.

## Auteur

Eya Ben Maiz - Projet réalisé avec Oumayma Bouaoija dans le cadre du PFA 2GII (ENET'Com Sfax)
