import pandas as pd
import numpy as np

# Parametro della normale
sigma = 2.17

# Caricamento del file CSV
df = pd.read_csv("posterior_top0_5perc_db.csv")  # Assicurati che sia nella stessa directory o fornisci il path completo

# Calcolo dei pesi con kernel gaussiano centrato in 0
df["peso"] = np.exp(- df["distanza"]**2 / (2 * sigma**2))

# Salvataggio del nuovo CSV
df[["x", "theta", "distanza", "peso"]].to_csv("posterior_top0_5perc_db_pesi.csv", index=False)

print("File generato con successo.")
