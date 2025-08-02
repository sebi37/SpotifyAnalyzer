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

#print("Erste Einträge im Datensatz:")
#print(df.head())

#print("\nSpalten im Datensatz:")
#print(df.columns)

#filtere nur Songs mit positivem Tempo
df = df[df["tempo"] > 0]

#anzahl der Testdaten in df
print(f"\nAnzahl der Testdaten: {len(df)}")

# Scatterplot Tempo vs Energy, farbig nach Genre
if "track_genre" in df.columns:
    genres = df["track_genre"].astype(str)
    unique_genres = genres.unique()
    cmap = plt.colormaps.get_cmap("tab20")
    genre_color_map = {g: cmap(i % cmap.N) for i, g in enumerate(unique_genres)}

    plt.figure(figsize=(10, 10))
    for g in unique_genres:
        mask = genres == g
        plt.scatter(df.loc[mask, "tempo"], df.loc[mask, "energy"] * 100,
                    color=genre_color_map[g], label=g, alpha=0.6, s=10)
    plt.xlabel("Tempo (BPM)")
    plt.ylabel("Energy (0–100)")
    plt.title("Tempo vs Energy nach Genre")
    plt.grid(True)
    plt.xlim(0, 250)
    plt.ylim(0, 100)
    plt.legend(markerscale=2, fontsize=9, loc="best", ncol=2)
    plt.tight_layout()
    plt.show()

    # 2. Abbildung: Genre-Gruppen (Oberkategorien)
    print("\nAbbildung 2: Tempo vs Energy nach Genre-Gruppen")
    # Beispielhafte Gruppierung, kann angepasst werden
    genre_groups = {
        "Pop": ["pop", "dance pop", "indie pop", "electropop", "acoustic pop", "pop rock"],
        "Rock": ["rock", "classic rock", "hard rock", "indie rock", "punk rock", "folk rock"],
        "Electronic": ["electronic", "edm", "house", "techno", "trance", "electro", "dubstep"],
        "Hip-Hop": ["hip hop", "rap", "trap", "german hip hop", "hip hop deutsch"],
        "Jazz": ["jazz", "vocal jazz", "jazz funk", "jazz fusion"],
        "Classical": ["classical", "modern classical", "orchestral", "piano"],
        "Folk": ["folk", "indie folk", "folk rock", "americana"],
        "Other": []
    }
    def map_group(genre):
        for group, members in genre_groups.items():
            if genre.lower() in members:
                return group
        return "Other"
    df["genre_group"] = df["track_genre"].apply(map_group)
    group_colors = plt.colormaps.get_cmap("Set2")
    unique_groups = df["genre_group"].unique()
    group_color_map = {g: group_colors(i % group_colors.N) for i, g in enumerate(unique_groups)}
    # Setze 'Other' auf helles Grau
    group_color_map["Other"] = "#eeeeee"
    plt.figure(figsize=(14, 7))
    for g in unique_groups:
        mask = df["genre_group"] == g
        # 'Other' nicht in der Legende anzeigen
        if g == "Other":
            plt.scatter(df.loc[mask, "tempo"], df.loc[mask, "energy"] * 100,
                        color=group_color_map[g], alpha=0.3, s=10)
        else:
            plt.scatter(df.loc[mask, "tempo"], df.loc[mask, "energy"] * 100,
                        color=group_color_map[g], label=g, alpha=0.6, s=10)
    plt.xlabel("Tempo (BPM)")
    plt.ylabel("Energy (0–100)")
    plt.title("Tempo vs Energy nach Genre-Gruppen")
    plt.grid(True)
    plt.xlim(0, 250)
    plt.ylim(0, 100)
    plt.legend(markerscale=2, fontsize=11, loc="best")
    plt.tight_layout()
    plt.show()

    # 3. Abbildung: Top 10 Genres + Andere
    print("\nAbbildung 3: Tempo vs Energy für Top 10 Genres + Andere")
    top_genres = df["track_genre"].value_counts().index[:10].tolist()
    def top10_or_other(genre):
        return genre if genre in top_genres else "Andere"
    df["genre_top10"] = df["track_genre"].apply(top10_or_other)
    top_colors = plt.colormaps.get_cmap("tab10")
    unique_top = df["genre_top10"].unique()
    top_color_map = {g: top_colors(i % top_colors.N) for i, g in enumerate(unique_top)}
    # Setze 'Andere' auf helles Grau
    top_color_map["Andere"] = "#eeeeee"
    plt.figure(figsize=(14, 7))
    for g in unique_top:
        mask = df["genre_top10"] == g
        # 'Andere' nicht in der Legende anzeigen
        if g == "Andere":
            plt.scatter(df.loc[mask, "tempo"], df.loc[mask, "energy"] * 100,
                        color=top_color_map[g], alpha=0.3, s=10)
        else:
            plt.scatter(df.loc[mask, "tempo"], df.loc[mask, "energy"] * 100,
                        color=top_color_map[g], label=g, alpha=0.6, s=10)
    plt.xlabel("Tempo (BPM)")
    plt.ylabel("Energy (0–100)")
    plt.title("Tempo vs Energy für Top 10 Genres + Andere")
    plt.grid(True)
    plt.xlim(0, 250)
    plt.ylim(0, 100)
    plt.legend(markerscale=2, fontsize=11, loc="best")
    plt.tight_layout()
    plt.show()
else:
    print("Keine track_genre-Spalte im Datensatz gefunden!")

