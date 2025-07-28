import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib

# Schritt 1: CSV laden
df = pd.read_csv('./data/datasetSpotify.csv')

# Schritt 2: Überblick
print("Erste Einträge im Datensatz:")
print(df.head())
print("\nSpalten im Datensatz:")
print(df.columns)

# Schritt 3: Features auswählen
feature_columns = [
    "valence", "energy", "danceability", "tempo", "loudness",
    "acousticness", "instrumentalness", "popularity"
]
feature_columns = [col for col in feature_columns if col in df.columns]

# Schritt 4: Mood-Spalte erzeugen
if "mood" not in df.columns:
    def valence_to_mood(x):
        if x > 0.6:
            return "Happy"
        elif x < 0.4:
            return "Sad"
        else:
            return "Neutral"
    df["mood"] = df["valence"].apply(valence_to_mood)

X = df[feature_columns]
y = df["mood"]

# Schritt 5: Daten splitten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Schritt 6: Modellvergleich
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} Klassifikationsbericht:")
    print(classification_report(y_test, y_pred))

# Schritt 7: Hyperparameter-Tuning für Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20]
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
grid.fit(X_train, y_train)
print("\nBestes RandomForest-Modell:", grid.best_params_)

# Schritt 8: Finale Auswertung mit bestem RF-Modell
best_rf = grid.best_estimator_
y_pred = best_rf.predict(X_test)
print("\nKlassifikationsbericht (RandomForest, getuned):")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=best_rf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_rf.classes_)
disp.plot()
plt.title("Konfusionsmatrix der Stimmungsklassifikation (RandomForest)")
plt.show()

# Schritt 9: Feature-Wichtigkeit visualisieren
importances = best_rf.feature_importances_
sns.barplot(x=importances, y=feature_columns)
plt.title("Feature-Wichtigkeit (RandomForest)")
plt.show()

# Schritt 10: Modell speichern
joblib.dump(best_rf, './data/best_rf_model.joblib')
print("Modell gespeichert unter './data/best_rf_model.joblib'")