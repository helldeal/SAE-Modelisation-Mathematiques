from pyexpat import model
from sklearn.linear_model import LogisticRegression
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
print(bcolors.OKCYAN + "nlp loaded" + bcolors.ENDC)
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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


# LogisticRegression
print(bcolors.OKBLUE + "LogisticRegression..." + bcolors.ENDC)
y = df['rating']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(multi_class='ovr')  # 'ovr' signifie One-vs-Rest
params = {'C': [0.01, 0.1, 1, 10]}
clf = GridSearchCV(model, params)
clf.fit(X_train, y_train)
print(bcolors.OKGREEN + "OK" + bcolors.ENDC)

# Évaluation de LogisticRegression
print(bcolors.OKBLUE + "Évaluation de LogisticRegression..." + bcolors.ENDC)
y_pred = clf.predict(X_test)
print("LogisticRegression Classification Report")
print(classification_report(y_test, y_pred, zero_division=1))
print(bcolors.OKGREEN + "OK" + bcolors.ENDC)