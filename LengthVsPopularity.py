

import pandas as pd
import matplotlib.pyplot as plt

# CSV-Datei laden
df = pd.read_csv("./data/datasetSpotify.csv")

# Dauer in Minuten berechnen
df["duration_min"] = df["duration_ms"] / 60000

# Streudiagramm: Dauer vs. Popularität
plt.figure(figsize=(10, 6))
plt.scatter(df["duration_min"], df["popularity"], alpha=0.4)
plt.title("Songlänge vs. Beliebtheit")
plt.xlabel("Dauer (Minuten)")
plt.ylabel("Popularität (0–100)")
plt.grid(True)
plt.show()

# Korrelation berechnen
correlation = df["duration_min"].corr(df["popularity"])
print(f"Korrelation zwischen Songlänge und Popularität: {correlation:.2f}")

# Durchschnittliche Popularität nach Längengruppen
df["length_group"] = pd.cut(df["duration_min"], bins=[0, 2, 3, 4, 5, 10],
                            labels=["<2", "2–3", "3–4", "4–5", "5+"])

grouped = df.groupby("length_group")["popularity"].mean()
print("\nDurchschnittliche Popularität nach Songlänge:")
print(grouped)

grouped.plot(kind="bar", title="Durchschnittliche Popularität pro Längengruppe")
plt.ylabel("Beliebtheit")
plt.xlabel("Songlänge (Minuten)")
plt.tight_layout()
plt.show()