from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot as plt
from data import *

# Définir la série temporelle à utiliser pour le modèle
ts = data['Index_diff']

# Ajuster le modèle ARIMA (p, d, q) -> (1,1,1) est un point de départ courant, ajustez selon les besoins
model = ARIMA(ts, order=(5, 0, 2))
model_fit = model.fit()

# Résumé du modèle
print(model_fit.summary())

# Faire des prédictions
predictions = model_fit.forecast(steps=30)  # prédire les 30 prochains jours

# Visualiser les prévisions
plt.figure(figsize=(10, 6))
plt.plot(ts, label='Valeurs Réelles')
plt.plot(predictions.index, predictions, label='Prédictions ARIMA', color='red')
plt.title('Prédiction de l\'Indice S&P 500 avec ARIMA')
plt.xlabel('Date')
plt.ylabel('Valeur Différenciée')
plt.legend()
plt.show()
