import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# Charger les données depuis un fichier CSV
data = pd.read_csv('sp500_index.csv')

# Convertir la colonne 'Date' en type datetime
data['Date'] = pd.to_datetime(data['Date'])

# Fixer la colonne Date comme index
data.set_index('Date', inplace=True)

# Afficher les premières lignes des données pour vérifier le format
print(data.head())

# Vérifier la stationnarité de la série temporelle avec le test ADF
result = adfuller(data['Index'])
print(f'Statistique ADF: {result[0]}')
print(f'p-value: {result[1]}')

# Si p-value > 0.05, la série n'est pas stationnaire
if result[1] > 0.05:
    data['Index_diff'] = data['Index'].diff().dropna()
else:
    data['Index_diff'] = data['Index']

# Supprimer les valeurs NaN générées par la différenciation
data.dropna(inplace=True)
