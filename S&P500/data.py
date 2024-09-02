import pandas as pd

# Charger les données financières depuis un fichier CSV
data = pd.read_csv('sp500_index.csv')

# Afficher les premières lignes des données
print(data.head())

# Supposons que les données contiennent les colonnes : Date, Open, High, Low, Close, Volume
# Convertir la colonne 'Date' en type datetime
data['Date'] = pd.to_datetime(data['Date'])

# Trier les données par date (au cas où elles ne le seraient pas)
data = data.sort_values('Date')

# Fixer la colonne Date comme index
data.set_index('Date', inplace=True)

# Afficher les informations sur les données
print(data.info())

# Créer des fonctionnalités supplémentaires si nécessaire, comme les moyennes mobiles
data['SMA_50'] = data['Index'].rolling(window=50).mean()  # Moyenne mobile sur 50 jours
data['SMA_200'] = data['Index'].rolling(window=200).mean()  # Moyenne mobile sur 200 jours

# Afficher les dernières lignes des données pour voir les nouvelles colonnes
print(data.tail())
