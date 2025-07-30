import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from sklearn.tree import DecisionTreeClassifier  # aktuell nur als Platzhalter
from collections import Counter
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
# Audio-Features einzeln abrufen für bessere Fehlerbehandlung

# --- Analyse mit datasetSpotify.csv ---
df = pd.read_csv("./data/datasetSpotify.csv")

print("Erste Einträge im Datensatz:")
print(df.head())

print("\nSpalten im Datensatz:")
print(df.columns)

# Scatterplot Tempo vs Energy, farbig nach Genre
if "track_genre" in df.columns:
    genres = df["track_genre"].astype(str)
    unique_genres = genres.unique()
    colors = plt.cm.get_cmap("tab20", len(unique_genres))
    genre_color_map = {g: colors(i) for i, g in enumerate(unique_genres)}

    plt.figure(figsize=(12, 7))
    for g in unique_genres:
        mask = genres == g
        plt.scatter(df.loc[mask, "tempo"], df.loc[mask, "energy"],
                    color=genre_color_map[g], label=g, alpha=0.6, s=10)
    plt.xlabel("Tempo (BPM)")
    plt.ylabel("Energy (0–1)")
    plt.title("Tempo vs Energy nach Genre")
    plt.grid(True)
    plt.legend(markerscale=2, fontsize=9, loc="best", ncol=2)
    plt.tight_layout()
    plt.show()
else:
    print("Keine track_genre-Spalte im Datensatz gefunden!")

