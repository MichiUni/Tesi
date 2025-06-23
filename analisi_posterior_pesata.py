import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carica il file con pesi
df = pd.read_csv("posterior_top0_5perc_db_pesi.csv")

# Crea istogramma pesato
plt.figure(figsize=(10, 6))
#sns.histplot(df["theta"], bins=100, kde=True, color="dodgerblue", edgecolor="black")
plt.hist(df["theta"], bins=100, weights=df["peso"], color='steelblue', edgecolor='black', density=True)

# Etichette e titolo
plt.xlabel("θ (posizione del danno)")
plt.ylabel("Densità pesata")
plt.title("Distribuzione pesata di θ (kernel ABC)")

# Linea verticale opzionale sulla posizione reale del danno
plt.axvline(x=2423.8, color='red', linestyle='--', label="θ vero (2423.8)")
plt.axvline(df["theta"].mean(), color="blue", linestyle="--", label="Media posterior")
plt.legend()

# Mostra il grafico
plt.tight_layout()
plt.savefig("posterior_bayesiana_pesata.png")
plt.close()