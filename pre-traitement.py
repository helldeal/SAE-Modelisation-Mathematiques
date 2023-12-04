import spacy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import scikit_learn
import json

data_list = []

# Ouvrez le fichier JSON en mode lecture
with open('goodreads_reviews_spoiler.json', 'r') as fichier_json:
    for ligne in fichier_json:
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

df = df[['review_sentences', 'rating']]

data_json = df.to_dict(orient='records')

# Écrire les données JSON dans un fichier
with open('resultat.json', 'w') as fichier_resultat:
    json.dump(data_json, fichier_resultat, indent=4)