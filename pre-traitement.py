import json
import pandas as pd
from tqdm import tqdm  # Importez tqdm pour la barre de progression

data_list = []

# Ouvrez le fichier JSON en mode lecture
with open('goodreads_reviews_spoiler.json', 'r') as fichier_json:
    lines = fichier_json.readlines()  # Lisez toutes les lignes du fichier en une seule opération
    for ligne in tqdm(lines, desc='Chargement des données'):
        try:
            # Chargez l'objet JSON de la ligne actuelle
            objet_json = json.loads(ligne)
            data_list.append(objet_json)
        except json.JSONDecodeError:
            # Ignorer les lignes vides ou mal formatées
            pass

# Utilisez json_normalize pour aplatir les données JSON
df = pd.json_normalize(data_list)

df = df[df['has_spoiler'] == True]

data_json = df.to_dict(orient='records')

# Écrire les données JSON dans un fichier
with open('onlyspoilers.json', 'w') as fichier_resultat:
    json.dump(data_json, fichier_resultat, indent=4)
