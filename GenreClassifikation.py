

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Datensatz laden
df = pd.read_csv("./data/datasetSpotify.csv")

# Nur die Top 15 Genres verwenden
top_genres = df['track_genre'].value_counts().nlargest(15).index
df = df[df['track_genre'].isin(top_genres)]

# Relevante Features und Zielspalte
features = ["danceability", "energy", "loudness", "speechiness",
            "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
target = "track_genre"

# Entferne Zeilen mit fehlenden Werten
df = df.dropna(subset=features + [target])

# Trainings- und Testdaten
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Klassifikationsmodell trainieren
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Feature-Wichtigkeit ausgeben
importances = model.feature_importances_
print("\n🔍 Wichtigste Audiofeatures:")
for feat, score in zip(features, importances):
    print(f"{feat}: {score:.4f}")

# Vorhersage und Auswertung
y_pred = model.predict(X_test)
print("🎯 Klassifikationsbericht:")
print(classification_report(y_test, y_pred))

# Konfusionsmatrix anzeigen
plt.figure(figsize=(12, 6))
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
sns.heatmap(cm, annot=False, xticklabels=model.classes_, yticklabels=model.classes_, cmap="Blues")
plt.title("Konfusionsmatrix der Genreklassifikation")
plt.xlabel("Vorhergesagt")
plt.ylabel("Tatsächlich")
plt.tight_layout()
plt.show()