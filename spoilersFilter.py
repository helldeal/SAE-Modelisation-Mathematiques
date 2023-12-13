import json
import pandas as pd


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def spoilersFilter():
    print(bcolors.OKBLUE + "Chargment des données..." + bcolors.ENDC)
    df = pd.read_json('onlyspoilers.json')
    print(bcolors.OKGREEN + "OK" + bcolors.ENDC)

    print(bcolors.OKBLUE + "Filtrage..." + bcolors.ENDC)
    df = df.drop_duplicates(subset=['user_id', 'book_id'])
    df['full_text'] = df['review_sentences'].apply(lambda sentences: ' '.join([sentence[1] for sentence in sentences]))
    df['label'] = df['rating'].apply(lambda x: 1 if x >= 2.5 else 0)  # 1 pour favorable, 0 pour défavorable
    df = df[['rating', 'label','full_text']]
    df = df.dropna(subset=['rating', 'label', 'full_text'])
    print(bcolors.OKGREEN + "OK" + bcolors.ENDC)

    print(bcolors.OKBLUE + "Ecriture dans data.json..." + bcolors.ENDC)
    data_json = df.to_dict(orient='records')
    with open('data.json', 'w') as fichier_resultat:
        json.dump(data_json, fichier_resultat, indent=4)
    print(bcolors.OKGREEN + "OK" + bcolors.ENDC)