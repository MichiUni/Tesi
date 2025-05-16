import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("posterior_bayesiano.csv")

plt.figure()
plt.hist(df["theta"], bins=50)
plt.xlabel("theta")
plt.ylabel("frequenza")
plt.title("Distribuzione theta")
plt.savefig("frequenza_theta.png")

plt.figure()
plt.hist(df["x"], bins=50)
plt.xlabel("x")
plt.ylabel("frequenza")
plt.title("Distribuzione x")
plt.savefig("frequenza_x.png")

print("✅ Grafici salvati.")
