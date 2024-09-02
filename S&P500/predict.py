from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from data import *

# Déplacer les valeurs de l'Index pour créer une colonne 'Future_Index'
data['Future_Index'] = data['Index'].shift(-1)

# Supprimer les lignes avec des valeurs manquantes (générées par le décalage et les moyennes mobiles)
data.dropna(inplace=True)

# Définir les caractéristiques (X) et la cible (y)
X = data[['Index', 'SMA_50', 'SMA_200']]
y = data['Future_Index']

# Diviser les données en ensembles d'entraînement et de test (sans mélanger les dates)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Créer et entraîner un modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

# Prédire les valeurs sur l'ensemble de test
y_pred = model.predict(X_test)

# Calculer l'erreur quadratique moyenne pour évaluer le modèle
mse = mean_squared_error(y_test, y_pred)
print(f'Erreur quadratique moyenne : {mse:.2f}')
