import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analisi_posterior(file_csv="posterior_bayesiano.csv"):
    # Carica il posterior
    df = pd.read_csv(file_csv)

    # Controllo dati
    print("✅ Posterior caricato:")
    print(df.head())

    # STATISTICHE DESCRITTIVE
    print("\n📊 Statistiche descrittive:")
    print(df.describe())

    # Istogramma delle accettazioni
    # plt.figure(figsize=(8, 4))
    # sns.histplot(df["accettazioni"], bins=30, kde=False)
    # plt.title("Distribuzione delle accettazioni")
    # plt.xlabel("Numero di accettazioni")
    # plt.ylabel("Frequenza")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("istogramma_accettazioni.png")
    # plt.close()

    # Distribuzione Theta Accettati
    # Filtro: solo righe con accettazioni > 0
    df_accettati = df[df["accettazioni"] > 0]
    # Plot distribuzione θ
    plt.figure(figsize=(10, 5))
    sns.histplot(df_accettati["theta"], bins=50, kde=True, color="royalblue", edgecolor="black")
    plt.title("Distribuzione di θ accettati (accettazioni > 0)")
    plt.xlabel("θ (posizione del danno)")
    plt.ylabel("Frequenza")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("distribuzione_theta_accettati.png")
    plt.show()

    # Scatter plot: theta vs accettazioni
    plt.figure(figsize=(8, 5))
    plt.scatter(df["theta"], df["accettazioni"], alpha=0.5, s=10)
    plt.title("Theta vs Numero di accettazioni")
    plt.xlabel("θ")
    plt.ylabel("Accettazioni")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("scatter_theta_accettazioni.png")
    plt.close()

    # Scatter plot: x vs accettazioni
    # plt.figure(figsize=(8, 5))
    # plt.scatter(df["x"], df["accettazioni"], alpha=0.5, s=10)
    # plt.title("x vs Numero di accettazioni")
    # plt.xlabel("x")
    # plt.ylabel("Accettazioni")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("scatter_x_accettazioni.png")
    # plt.close()

    # Heatmap: media accettazioni su griglia
    grid = df.copy()
    grid["x_bin"] = pd.cut(grid["x"], bins=30)
    grid["theta_bin"] = pd.cut(grid["theta"], bins=30)
    pivot = grid.pivot_table(index="x_bin", columns="theta_bin", values="accettazioni", aggfunc="mean")

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, cmap="viridis", cbar_kws={'label': 'Accettazioni medie'})
    plt.title("Heatmap accettazioni medie (x vs θ)")
    plt.xlabel("θ (binned)")
    plt.ylabel("x (binned)")
    plt.tight_layout()
    plt.savefig("heatmap_accettazioni.png")
    plt.close()

    # Boxplot accettazioni
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df["accettazioni"])
    plt.title("Boxplot delle accettazioni")
    plt.xlabel("Accettazioni")
    plt.tight_layout()
    plt.savefig("boxplot_accettazioni.png")
    plt.close()

    # Jointplot θ vs x (per accettazioni > 0)
    df_pos = df[df["accettazioni"] > 0]
    if not df_pos.empty:
        jp = sns.jointplot(data=df_pos, x="theta", y="x", hue="accettazioni", palette="viridis", height=6, alpha=0.7)
        jp.fig.suptitle("Jointplot θ vs x (solo accettazioni > 0)", fontsize=12)
        jp.fig.tight_layout()
        jp.fig.subplots_adjust(top=0.95)
        jp.savefig("jointplot_theta_x_accettati.png")
        plt.close()

    # Zoom su θ ∈ [2000, 3000]
    df_zoom = df[(df["theta"] >= 2000) & (df["theta"] <= 3000)]
    plt.figure(figsize=(8, 5))
    plt.scatter(df_zoom["theta"], df_zoom["accettazioni"], alpha=0.5, s=10)
    plt.title("Zoom: θ ∈ [2000, 3000]")
    plt.xlabel("θ")
    plt.ylabel("Accettazioni")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("zoom_theta_2000_3000.png")
    plt.close()

    print("📈 Grafici generati e salvati.")

if __name__ == "__main__":
    analisi_posterior("posterior_eps10.csv")
