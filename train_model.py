"""
train_model.py — Entraînement XGBoost adapté à dataset_final.csv
-----------------------------------------------------------------
Ce script reproduit exactement le feature engineering de ton notebook
et sauvegarde les 3 fichiers nécessaires au backend :
  - model/xgboost_model.pkl
  - model/scaler.pkl
  - model/region_encoding.pkl

Usage :
    python train_model.py --data dataset_final.csv
"""

import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import os

# ── Arguments ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data", default="dataset_final.csv",
                    help="Chemin vers le fichier CSV")
parser.add_argument("--out", default="model",
                    help="Dossier de sortie des fichiers .pkl")
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

# ── 1. Chargement & nettoyage ─────────────────────────────────────────────────
print("Chargement du dataset…")
df = pd.read_csv(args.data)
df = df.drop_duplicates()
df = df[df['Production_spec_kWh_per_kWp'].between(0.5, 12)]
print(f"Lignes après nettoyage : {len(df):,}")

# ── 2. Feature engineering ────────────────────────────────────────────────────
df['Date']       = pd.to_datetime(df['Date'])
df['Mois']       = df['Date'].dt.month
df['Jour_annee'] = df['Date'].dt.dayofyear

df['Solar_x_Tmax']   = df['Solar_Wm2'] * df['Tmax']
df['Solar_net']       = df['Solar_Wm2'] * (1 - df['Precip_mm'].clip(0, 1))
df['Delta_T']         = df['Tmax'] - df['Tmin']
df['Temp_efficiency'] = df['Tavg'] / (df['Solar_Wm2'] + 0.001)
df['Cooling_effect']  = df['Wind_mps'] * df['Tavg']
df['Low_solar']       = (df['Solar_Wm2'] < 3).astype(int)
df['Solar_per_temp']  = df['Solar_Wm2'] / (df['Tavg'] + 1)
df['Mois_sin']        = np.sin(2 * np.pi * df['Mois'] / 12)
df['Mois_cos']        = np.cos(2 * np.pi * df['Mois'] / 12)

# Encodage type centrale (one-hot)
df = pd.get_dummies(df, columns=['Type_centrale'])

# Encodage région (mean target encoding) — sauvegardé pour l'inférence
t_mean = df.groupby('Region')['Production_kWh'].mean()
df['Region_encoded'] = df['Region'].map(t_mean)
joblib.dump(t_mean, os.path.join(args.out, 'region_encoding.pkl'))
df.drop(columns=['Region'], inplace=True)

print("Feature engineering terminé.")

# ── 3. Normalisation ──────────────────────────────────────────────────────────
col_a_normaliser = ['Capacite_kWp', 'Tavg', 'Tmax', 'Tmin',
                    'Precip_mm', 'Wind_mps', 'Solar_Wm2', 'Humidity']

scaler = MinMaxScaler()
df[col_a_normaliser] = scaler.fit_transform(df[col_a_normaliser])
joblib.dump(scaler, os.path.join(args.out, 'scaler.pkl'))
print("Scaler sauvegardé.")

# ── 4. Colonnes de features utilisées (ordre exact pour l'inférence) ──────────
# On exclut les colonnes non-features
EXCLUDE = ['Date', 'Production_kWh', 'Production_spec_kWh_per_kWp', 'Saison']
FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE]
joblib.dump(FEATURE_COLS, os.path.join(args.out, 'feature_cols.pkl'))
print(f"Features ({len(FEATURE_COLS)}) : {FEATURE_COLS}")

# ── 5. Train / test split ─────────────────────────────────────────────────────
np.random.seed(42)
test_indices = []

for mois in sorted(df['Date'].dt.month.unique()):
    dates_mois = df[df['Date'].dt.month == mois]['Date'].unique()
    n_test = max(1, int(len(dates_mois) * 0.2))
    dates_test_mois = np.random.choice(dates_mois, size=n_test, replace=False)
    idx = df[df['Date'].isin(dates_test_mois)].index
    test_indices.extend(idx)

test_indices  = list(set(test_indices))
train_indices = [i for i in df.index if i not in set(test_indices)]

x_train = df.loc[train_indices, FEATURE_COLS]
x_test  = df.loc[test_indices,  FEATURE_COLS]
y_train = df.loc[train_indices, 'Production_spec_kWh_per_kWp']
y_test  = df.loc[test_indices,  'Production_spec_kWh_per_kWp']

# Sauvegarder les dates du test set pour usage futur
dates_test = df.loc[test_indices, 'Date'].dt.strftime('%Y/%m/%d').unique()
pd.Series(dates_test).to_csv('dates_test_set.csv', index=False, header=['Date'])

print(f"Train : {len(x_train):,} lignes")
print(f"Test  : {len(x_test):,} lignes")
print(f"Dates test (uniques) : {len(dates_test)}")
# ── 6. Entraînement XGBoost ───────────────────────────────────────────────────
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.07,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
print("Entraînement en cours…")
model.fit(x_train, y_train)

joblib.dump(model, os.path.join(args.out, 'xgboost_model.pkl'))
print("✅ Modèle sauvegardé → model/xgboost_model.pkl")

# ── 7. Évaluation ─────────────────────────────────────────────────────────────
y_pred = model.predict(x_test)
print(f"\n{'='*40}")
print(f"R²  train : {model.score(x_train, y_train):.4f}")
print(f"R²  test  : {model.score(x_test, y_test):.4f}")
print(f"MAE       : {mean_absolute_error(y_test, y_pred):.4f} kWh/kWp")
print(f"RMSE      : {mean_squared_error(y_test, y_pred):.4f}")
print(f"{'='*40}")
print("\nTout est prêt. Lance maintenant : python app.py")
