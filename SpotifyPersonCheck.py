import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from dotenv import load_dotenv
import os

# .env-Datei laden
load_dotenv()

# Zugriff auf die Variablen
client_id = os.getenv("SPOTIPY_CLIENT_ID")
client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")

# Spotify API-Authentifizierung
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=client_id,
    client_secret=client_secret,
    redirect_uri="http://127.0.0.1:8888/callback",
    scope="user-top-read"
))

print("🎧 Deine meistgehörten Songs (letzte 4 Wochen):\n")

top_tracks = sp.current_user_top_tracks(limit=10, time_range='short_term')

for idx, item in enumerate(top_tracks['items']):
    name = item['name']
    artist = item['artists'][0]['name']
    print(f"{idx+1}. {name} – {artist}")

# Genre-Analyse der Top-Tracks
from collections import Counter

genre_counter = Counter()

for item in top_tracks['items']:
    artist_id = item['artists'][0]['id']
    artist_info = sp.artist(artist_id)
    genres = artist_info['genres']
    genre_counter.update(genres)

# Ausgabe der häufigsten Genres
print("\n🎼 Meistgehörte Genres:")
for genre, count in genre_counter.most_common(10):
    print(f"{genre}: {count}x")

# Grobe Genre-Zuordnung
übergenre_map = {
    "metal": "Metalhead",
    "death metal": "Metalhead",
    "black metal": "Metalhead",
    "rock": "Rocker",
    "hard rock": "Rocker",
    "punk": "Rocker",
    "edm": "Electro",
    "electronic": "Electro",
    "house": "Electro",
    "techno": "Electro",
    "dubstep": "Dubstep",
    "pop": "Pop-Liebhaber",
    "indie pop": "Indie",
    "synthpop": "Indie",
    "hip hop": "Hip-Hop Fan",
    "trap": "Hip-Hop Fan",
    "rap": "Hip-Hop Fan",
    "indie rock": "Indie",
    "alternative": "Indie"
}

übergenre_counter = Counter()

for genre, count in genre_counter.items():
    for key in übergenre_map:
        if key in genre:
            übergenre_counter[übergenre_map[key]] += count
            break

if übergenre_counter:
    top_typ = übergenre_counter.most_common(1)[0][0]
    print(f"\n🧠 Musikprofil: Du bist wahrscheinlich ein/e {top_typ}!")
else:
    print("Keine eindeutige Genre-Zuordnung gefunden.")
