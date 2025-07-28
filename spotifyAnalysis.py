# Spotify Mood Classification mit CSV und Machine Learning

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Schritt 1: CSV-Datei laden
df = pd.read_csv("./data/datasetSpotify.csv")  # Passe den Dateinamen ggf. an

# Schritt 2: Überblick über die Daten
print("Erste Einträge im Datensatz:")
print(df.head())
print("\nSpalten im Datensatz:")
print(df.columns)

# Schritt 3: Spalten auswählen
feature_columns = ["valence", "energy", "danceability", "tempo", "loudness"]

# Falls keine 'mood'-Spalte vorhanden ist, wird sie aus 'valence' abgeleitet
if "mood" not in df.columns:
    def valence_to_mood(x):
        if x > 0.6:
            return "Happy"
        elif x < 0.4:
            return "Sad"
        else:
            return "Neutral"
    df["mood"] = df["valence"].apply(valence_to_mood)

# Schritt 4: Eingabematrix & Zielvariable
X = df[feature_columns]
y = df["mood"]

# Schritt 5: Trainings- und Testdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Schritt 6: Modell trainieren
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Schritt 7: Vorhersagen & Auswertung
y_pred = model.predict(X_test)

print("\nKlassifikationsbericht:")
print(classification_report(y_test, y_pred))

# Schritt 8: Visualisierung der Konfusionsmatrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.title("Konfusionsmatrix der Stimmungsklassifikation")
plt.show()