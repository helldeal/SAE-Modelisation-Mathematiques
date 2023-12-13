from pyexpat import model
from sklearn.neighbors import KNeighborsClassifier
from preTraitement import preTraitement
from spoilersFilter import bcolors, spoilersFilter

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import spacy

#preTraitement()
#spoilersFilter()


print(bcolors.OKBLUE + "Lecture..." + bcolors.ENDC)
df = pd.read_json('data.json')
df = df.head(1000)
print(bcolors.OKGREEN + "OK" + bcolors.ENDC)

# Prétraitement avec spaCy
print(bcolors.OKBLUE + "Prétraitement avec spaCy..." + bcolors.ENDC)
nlp = spacy.load("en_core_web_sm")
def preprocess(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

df['processed_text'] = df['full_text'].apply(preprocess)
print(bcolors.OKGREEN + "OK" + bcolors.ENDC)

# Vectorisation
print(bcolors.OKBLUE + "Vectorisation..." + bcolors.ENDC)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['processed_text'])
y = df['label']
print(bcolors.OKGREEN + "OK" + bcolors.ENDC)

# Division des données
print(bcolors.OKBLUE + "Division des données..." + bcolors.ENDC)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(bcolors.OKGREEN + "OK" + bcolors.ENDC)

# KNN
print(bcolors.OKBLUE + "KNN..." + bcolors.ENDC)
knn_model = KNeighborsClassifier()
knn_params = {'n_neighbors': [3, 5, 7, 9]}
knn_clf = GridSearchCV(knn_model, knn_params)
knn_clf.fit(X_train, y_train)
print(bcolors.OKGREEN + "OK" + bcolors.ENDC)


# Évaluation de KNN
print(bcolors.OKBLUE + "Évaluation de KNN..." + bcolors.ENDC)
y_pred_knn = knn_clf.predict(X_test)
print("KNN Classification Report")
print(classification_report(y_test, y_pred_knn, zero_division=1))
print(bcolors.OKGREEN + "OK" + bcolors.ENDC)

def test_phrase(phrase):
    # Prétraitement
    processed_phrase = preprocess(phrase)  # Utilisez votre fonction de prétraitement ici

    # Vectorisation
    vectorized_phrase = vectorizer.transform([processed_phrase])

    # Faire une prédiction
    prediction = knn_clf.predict(vectorized_phrase)

    # Interpréter le résultat
    if prediction[0] == 1:
        return "Avis favorable"
    else:
        return "Avis défavorable"

# Test d'une phrase
phrase = "I am extremely disappointed with this product. It did not meet my expectations at all and failed to deliver on its promises. The quality was poor, and it broke shortly after purchase. Customer service was less than helpful, barely answering my questions and offering no satisfactory solutions. I definitely do not recommend this item to anyone. It's a complete waste of money and time."
result = test_phrase(phrase)
print("Résultat de la classification :", result)