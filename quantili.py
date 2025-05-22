import pandas as pd

# Carica il posterior
df = pd.read_csv("posterior_eps10.csv")

# Filtra solo le righe accettate
df_accettati = df[df["accettazioni"] > 0]

# Calcola quantili su theta e x
quantili_theta = df_accettati["theta"].quantile([0, 0.25, 0.5, 0.75, 1.0])
quantili_x = df_accettati["x"].quantile([0, 0.25, 0.5, 0.75, 1.0])

print("Quantili di θ (solo righe accettate):")
print(quantili_theta)

print("\nQuantili di x (solo righe accettate):")
print(quantili_x)
