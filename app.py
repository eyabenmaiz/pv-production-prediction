"""
app.py — Backend Flask avec OpenWeatherMap + PVGIS
---------------------------------------------------
Routes :
    POST /predict       → Prédiction de production PV
    GET  /health        → Vérification état du serveur
    GET  /meta          → Valeurs possibles des champs catégoriels
    GET  /test          → Test de connexion
"""
from dotenv import load_dotenv
load_dotenv()
import os
import numpy as np
import pandas as pd
import joblib
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES ARTEFACTS
# ══════════════════════════════════════════════════════════════════════════════
MODEL_DIR = os.environ.get("MODEL_DIR", "model")

def load_artifact(name):
    path = os.path.join(MODEL_DIR, name)
    try:
        obj = joblib.load(path)
        print(f"✅ Chargé : {path}")
        return obj
    except FileNotFoundError:
        print(f"⚠️  Introuvable : {path}")
        return None

model           = load_artifact("xgboost_model.pkl")
scaler          = load_artifact("scaler.pkl")
region_encoding = load_artifact("region_encoding.pkl")
feature_cols    = load_artifact("feature_cols.pkl")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════
TYPE_CENTRALE_VALUES = ["Résidentiel", "Commercial", "Industriel"]
REGIONS_DISPONIBLES  = list(region_encoding.index) if region_encoding is not None else []
COL_A_NORMALISER     = ['Capacite_kWp', 'Tavg', 'Tmax', 'Tmin',
                         'Precip_mm', 'Wind_mps', 'Solar_Wm2', 'Humidity']

OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")

if not OPENWEATHER_API_KEY:
    print("ATTENTION : OPENWEATHER_API_KEY non définie dans l'environnement")

# ── Coordonnées GPS des régions tunisiennes ───────────────────────────────────
REGION_COORDS = {
    'Tunisie'                      : (33.80,  9.50),
    'Tunisie/Ariana'               : (36.86, 10.19),
    'Tunisie/Ariana/Ariana Médina' : (36.86, 10.19),
    'Tunisie/Ariana/Soukra'        : (36.89, 10.23),
    'Tunisie/Bizerte'              : (37.27,  9.87),
    'Tunisie/Gabès/Gabès Médina'   : (33.88, 10.10),
    'Tunisie/Gabès/Gabès Sud'      : (33.82, 10.08),
    'Tunisie/Mahdia'               : (35.50, 11.06),
    'Tunisie/Monastir'             : (35.77, 10.83),
    'Tunisie/Médenine'             : (33.35, 10.50),
    'Tunisie/Médenine/Zarzis'      : (33.50, 11.11),
    'Tunisie/Nabeul'               : (36.45, 10.73),
    'Tunisie/Sfax'                 : (34.74, 10.76),
    'Tunisie/Sfax/Sakiet Eddaier'  : (34.78, 10.77),
    'Tunisie/Sfax/Sakiet Ezzit'    : (34.76, 10.76),
    'Tunisie/Sfax/Sfax Médina'     : (34.74, 10.76),
    'Tunisie/Sfax/Sfax Sud'        : (34.70, 10.75),
    'Tunisie/Sfax/Skhira'          : (34.30, 10.07),
    'Tunisie/Sousse'               : (35.82, 10.64),
    'Tunisie/Tataouine/Ghomrassen' : (32.45, 10.17),
    'Tunisie/Tunis'                : (36.81, 10.18),
    'Tunisie/Tunis/La Goulette'    : (36.81, 10.30),
}

# ── Irradiance moyenne mensuelle par région (fallback si PVGIS indisponible) ──
# Valeurs en kWh/m²/jour — moyennes historiques Tunisie
SOLAR_FALLBACK_MONTHLY = {
    1: 3.2, 2: 4.0, 3: 5.2, 4: 6.1, 5: 7.0,
    6: 7.8, 7: 8.2, 8: 7.6, 9: 6.3, 10: 4.8,
    11: 3.5, 12: 2.9
}

# ══════════════════════════════════════════════════════════════════════════════
# FONCTION PVGIS — Irradiance solaire réelle
# ══════════════════════════════════════════════════════════════════════════════
def get_solar_pvgis(lat, lon, month, day):
    """
    Récupère l'irradiance solaire journalière via l'API PVGIS (EU Commission).
    Gratuite, sans clé API.
    Retourne la valeur en kWh/m²/jour.
    Fallback : moyenne mensuelle historique si API indisponible.
    """
    try:
        url = (
            f"https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
            f"?lat={lat}&lon={lon}"
            f"&startyear=2020&endyear=2020"
            f"&pvcalculation=0"
            f"&outputformat=json"
            f"&browser=0"
        )
        response = requests.get(url, timeout=10)
        data     = response.json()

        outputs  = data['outputs']['hourly']
        df_pvgis = pd.DataFrame(outputs)
        df_pvgis['time']  = pd.to_datetime(df_pvgis['time'], format='%Y%m%d:%H%M')
        df_pvgis['month'] = df_pvgis['time'].dt.month
        df_pvgis['day']   = df_pvgis['time'].dt.day

        day_data = df_pvgis[
            (df_pvgis['month'] == month) &
            (df_pvgis['day']   == day)
        ]

        if day_data.empty:
            print(f"⚠️ PVGIS : pas de données pour mois={month} jour={day} → fallback mensuel")
            return SOLAR_FALLBACK_MONTHLY.get(month, 6.0)

        # G(i) en W/m² → somme sur 24h → kWh/m²/jour
        solar_daily = day_data['G(i)'].sum() / 1000.0
        print(f"✅ PVGIS : Solar_Wm2 = {solar_daily:.3f} kWh/m²/jour")
        return round(solar_daily, 3)

    except Exception as e:
        print(f"⚠️ PVGIS indisponible ({e}) → fallback mensuel")
        return SOLAR_FALLBACK_MONTHLY.get(month, 6.0)

# ══════════════════════════════════════════════════════════════════════════════
# FONCTION OPENWEATHERMAP — Données météo
# ══════════════════════════════════════════════════════════════════════════════
def get_forecast(city, region=None, date=None):
    """
    Récupère les prévisions météo via OpenWeatherMap
    et l'irradiance solaire via PVGIS.
    """
    # ── OpenWeatherMap ──
    url = (
        f"http://api.openweathermap.org/data/2.5/forecast"
        f"?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    )
    response     = requests.get(url, timeout=5)
    data_owm     = response.json()

    if response.status_code != 200:
        raise ValueError(f"OpenWeatherMap erreur : {data_owm.get('message', 'inconnue')}")

    # Prendre la prévision J+1 (index 8 = +24h avec pas de 3h)
    idx          = 8 if len(data_owm["list"]) > 8 else 0
    next_weather = data_owm["list"][idx]

    # ── PVGIS pour irradiance ──
    dt       = pd.to_datetime(date) if date else pd.Timestamp.today() + pd.Timedelta(days=1)
    lat, lon = REGION_COORDS.get(region, (33.8, 9.5))
    solar    = get_solar_pvgis(lat, lon, dt.month, dt.day)

    return {
        "Tavg"     : round(next_weather["main"]["temp"],     2),
        "Tmax"     : round(next_weather["main"]["temp_max"], 2),
        "Tmin"     : round(next_weather["main"]["temp_min"], 2),
        "Humidity" : round(next_weather["main"]["humidity"], 2),
        "Wind_mps" : round(next_weather["wind"]["speed"],    2),
        "Precip_mm": round(next_weather.get("rain", {}).get("3h", 0.0), 2),
        "Solar_Wm2": solar
    }

# ══════════════════════════════════════════════════════════════════════════════
# CONSTRUCTION DES FEATURES
# ══════════════════════════════════════════════════════════════════════════════
def build_features(data: dict) -> pd.DataFrame:
    """
    Reconstruit exactement le vecteur de features utilisé à l'entraînement.
    """
    date = pd.to_datetime(data.get("date", pd.Timestamp.today()))

    row = {
        "Capacite_kWp" : float(data["Capacite_kWp"]),
        "Tavg"         : float(data["Tavg"]),
        "Tmax"         : float(data["Tmax"]),
        "Tmin"         : float(data["Tmin"]),
        "Precip_mm"    : float(data.get("Precip_mm", 0)),
        "Wind_mps"     : float(data["Wind_mps"]),
        "Solar_Wm2"    : float(data["Solar_Wm2"]),
        "Humidity"     : float(data["Humidity"]),
        "Mois"         : date.month,
        "Jour_annee"   : date.day_of_year,
    }

    # Features dérivées
    row["Solar_x_Tmax"]   = row["Solar_Wm2"] * row["Tmax"]
    row["Solar_net"]       = row["Solar_Wm2"] * (1 - min(row["Precip_mm"], 1))
    row["Delta_T"]         = row["Tmax"] - row["Tmin"]
    row["Temp_efficiency"] = row["Tavg"] / (row["Solar_Wm2"] + 0.001)
    row["Cooling_effect"]  = row["Wind_mps"] * row["Tavg"]
    row["Low_solar"]       = int(row["Solar_Wm2"] < 3)
    row["Solar_per_temp"]  = row["Solar_Wm2"] / (row["Tavg"] + 1)
    row["Mois_sin"]        = np.sin(2 * np.pi * row["Mois"] / 12)
    row["Mois_cos"]        = np.cos(2 * np.pi * row["Mois"] / 12)

    # One-hot Type_centrale
    tc = data.get("Type_centrale", "Résidentiel")
    for t in TYPE_CENTRALE_VALUES:
        row[f"Type_centrale_{t}"] = int(tc == t)

    # Region encoded
    region = data.get("Region", None)
    if region and region_encoding is not None and region in region_encoding.index:
        row["Region_encoded"] = float(region_encoding[region])
    elif region_encoding is not None:
        row["Region_encoded"] = float(region_encoding.mean())
    else:
        row["Region_encoded"] = 0.0

    df_row = pd.DataFrame([row])

    # Normalisation
    if scaler is not None:
        df_row[COL_A_NORMALISER] = scaler.transform(df_row[COL_A_NORMALISER])

    # Réordonner selon l'ordre exact de l'entraînement
    if feature_cols is not None:
        for col in feature_cols:
            if col not in df_row.columns:
                df_row[col] = 0
        df_row = df_row[feature_cols]

    return df_row

# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/predict", methods=["POST"])
def predict():
    """
    Corps JSON attendu :
    {
        "date"          : "2025-06-15",        (optionnel)
        "Capacite_kWp"  : 10.0,
        "Tavg"          : 28.5,
        "Tmax"          : 33.0,
        "Tmin"          : 24.0,
        "Precip_mm"     : 0.0,                 (optionnel)
        "Wind_mps"      : 3.2,
        "Solar_Wm2"     : 6.5,
        "Humidity"      : 55.0,
        "Type_centrale" : "Résidentiel",
        "Region"        : "Tunisie/Sousse",    (optionnel)
        "use_forecast"  : true,                (optionnel — active OpenWeatherMap + PVGIS)
        "city"          : "Sousse"             (requis si use_forecast = true)
    }
    """
    data = request.get_json(force=True)

    # ── Mode automatique OpenWeatherMap + PVGIS ──
    use_forecast = data.get("use_forecast", False)
    if use_forecast:
        city   = data.get("city", "Tunis")
        region = data.get("Region", None)
        date   = data.get("date", None)
        try:
            weather = get_forecast(city, region=region, date=date)
            data.update(weather)
            print(f"📡 Météo auto récupérée pour {city} : {weather}")
        except Exception as e:
            return jsonify({
                "status" : "error",
                "message": f"Erreur récupération météo : {e}"
            }), 500

    # ── Validation champs obligatoires ──
    required = ["Capacite_kWp", "Tavg", "Tmax", "Tmin",
                "Wind_mps", "Solar_Wm2", "Humidity"]
    missing  = [f for f in required if f not in data]
    if missing:
        return jsonify({
            "status" : "error",
            "message": f"Champs manquants : {', '.join(missing)}"
        }), 400

    # ── Validation des plages ──
    checks = [
        ("Tavg",        -30,   60,    data.get("Tavg")),
        ("Tmax",        -30,   65,    data.get("Tmax")),
        ("Tmin",        -30,   55,    data.get("Tmin")),
        ("Solar_Wm2",     0,   25,    data.get("Solar_Wm2")),
        ("Humidity",      0,  100,    data.get("Humidity")),
        ("Wind_mps",      0,  150,    data.get("Wind_mps")),
        ("Capacite_kWp",  0.1, 10000, data.get("Capacite_kWp")),
    ]
    for name, lo, hi, val in checks:
        try:
            v = float(val)
            if not (lo <= v <= hi):
                return jsonify({
                    "status" : "error",
                    "message": f"{name} hors plage [{lo}, {hi}]"
                }), 422
        except (TypeError, ValueError):
            return jsonify({
                "status" : "error",
                "message": f"Valeur invalide pour {name}"
            }), 400

    # ── Validation Type_centrale ──
    tc = data.get("Type_centrale", "Résidentiel")
    if tc not in TYPE_CENTRALE_VALUES:
        return jsonify({
            "status" : "error",
            "message": f"Type_centrale invalide. Valeurs acceptées : {TYPE_CENTRALE_VALUES}"
        }), 422

    # ── Construction des features ──
    try:
        df_input = build_features(data)
    except Exception as e:
        return jsonify({
            "status" : "error",
            "message": f"Erreur construction features : {e}"
        }), 500

    # ── Prédiction ──
    try:
        if model is None:
            return jsonify({
                "status" : "error",
                "message": "Modèle non chargé."
            }), 503

        prediction = float(model.predict(df_input)[0])
        prediction = round(max(prediction, 0), 4)

    except Exception as e:
        return jsonify({
            "status" : "error",
            "message": f"Erreur prédiction : {e}"
        }), 500

    # ── Calcul production totale ──
    capacite          = float(data["Capacite_kWp"])
    production_totale = round(prediction * capacite, 2)

    # ── Simulation multi-capacités ──
    simulations = [
        {"capacite_kWp": cap, "production_kWh": round(prediction * cap, 2)}
        for cap in [1, 3, 5, 10, 20, 50, 100]
    ]

    return jsonify({
        "status"                      : "success",
        "production_spec_kWh_per_kWp" : prediction,
        "production_totale_kWh"       : production_totale,
        "capacite_kWp"                : capacite,
        "type_centrale"               : tc,
        "region"                      : data.get("Region", "Non spécifiée"),
        "solar_wm2_utilise"           : round(float(data.get("Solar_Wm2", 0)), 3),
        "source_solar"                : "PVGIS" if use_forecast else "Manuel",
        "simulations"                 : simulations,
        "model_used"                  : "XGBoost"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status"         : "ok",
        "model_loaded"   : model is not None,
        "scaler_loaded"  : scaler is not None,
        "regions_count"  : len(REGIONS_DISPONIBLES),
        "features_count" : len(feature_cols) if feature_cols else 0
    })


@app.route("/meta", methods=["GET"])
def meta():
    return jsonify({
        "types_centrale" : TYPE_CENTRALE_VALUES,
        "regions"        : REGIONS_DISPONIBLES,
        "features"       : feature_cols if feature_cols else [],
        "region_coords"  : {k: {"lat": v[0], "lon": v[1]}
                            for k, v in REGION_COORDS.items()}
    })


@app.route("/test", methods=["GET"])
def test():
    return jsonify({"status": "ok", "message": "Serveur opérationnel"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
