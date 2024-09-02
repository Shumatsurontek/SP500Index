import matplotlib.pyplot as plt

# Données de base
names = ['John', 'Anna', 'Peter', 'Linda']
ages = [28, 24, 35, 32]

# Créer un graphique à barres
plt.bar(names, ages)
plt.xlabel('Names')
plt.ylabel('Ages')
plt.title('Ages of Individuals')
plt.show()
