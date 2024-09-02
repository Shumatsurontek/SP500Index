import numpy as np
import matplotlib.pyplot as plt
from predict import *

# Calculer les intervalles de confiance
y_pred_upper = y_pred + 1.96 * np.std(y_pred - y_test)
y_pred_lower = y_pred - 1.96 * np.std(y_pred - y_test)

# Visualiser les résultats
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Valeurs Réelles', color='blue')
plt.plot(y_test.index, y_pred, label='Prédictions', color='red')
plt.fill_between(y_test.index, y_pred_lower, y_pred_upper, color='gray', alpha=0.2, label='Intervalle de Confiance à 95%')
plt.legend()
plt.title('Prédiction de l\'Indice S&P 500 avec Intervalle de Confiance')
plt.show()
