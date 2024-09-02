import pandas as pd

# Création d'un DataFrame
data = {
    'Name': ['John', 'Anna', 'Peter', 'Linda'],
    'Age': [28, 24, 35, 32],
    'City': ['New York', 'Paris', 'Berlin', 'London']
}

df = pd.DataFrame(data)

# Afficher le DataFrame
print(df)

# Filtrer les données : sélectionner les personnes de moins de 30 ans
filtered_df = df[df['Age'] < 30]
print(filtered_df)

# Ajouter une nouvelle colonne avec des données calculées
df['Age in 10 Years'] = df['Age'] + 10
print(df)

# Résumé statistique des données
print(df.describe())

# Grouper les données par ville et compter les occurrences
city_counts = df.groupby('City').size()
print(city_counts)
