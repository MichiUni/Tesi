import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def analisi_posterior_pmc(file_csv="posterior_samples_pmc.csv", theta_true=2423.8):
    df = pd.read_csv(file_csv)

    if not all(col in df.columns for col in ["theta", "d_elle"]):
        raise ValueError("Il file CSV deve contenere almeno le colonne: 'theta' e 'd_elle'.")

    # ========================
    # 1. Posterior distribution di θ
    # ========================
    plt.figure(figsize=(10, 5))
    sns.histplot(df["theta"], bins=100, kde=True, color="dodgerblue", edgecolor="black")
    plt.axvline(theta_true, color="red", linestyle="--", label="θ vero")
    plt.axvline(df["theta"].mean(), color="blue", linestyle="--", label="Media posterior")
    plt.title("Distribuzione a posteriori di θ (location del danno)")
    plt.xlabel("θ (mm)")
    plt.ylabel("Densità")
    plt.legend()
    plt.tight_layout()
    plt.savefig("posterior_theta_pmc.png")
    plt.close()

    # ========================
    # 2. Jointplot θ vs d_elle
    # ========================
    jp = sns.jointplot(data=df, x="theta", y="d_elle", color="seagreen", height=6, alpha=0.6)
    jp.fig.suptitle("Jointplot θ vs d_elle", fontsize=12)
    jp.fig.tight_layout()
    jp.fig.subplots_adjust(top=0.95)
    jp.savefig("jointplot_theta_delle.png")
    plt.close()

    # ========================
    # 3. Zoom su θ ∈ [2000, 3000]
    # ========================
    df_zoom = df[(df["theta"] >= 2000) & (df["theta"] <= 3000)]
    plt.figure(figsize=(8, 5))
    plt.scatter(df_zoom["theta"], df_zoom["d_elle"], alpha=0.5, s=10, color="darkgreen")
    plt.axvline(theta_true, color="red", linestyle="--", label="θ vero")
    plt.title("Zoom: θ ∈ [2000, 3000]")
    plt.xlabel("θ")
    plt.ylabel("d_elle")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("zoom_theta_2000_3000.png")
    plt.close()

    # ========================
    # 4. Heatmap densità congiunta (conteggi)
    # ========================
    grid = df.copy()
    grid["theta_bin"] = pd.cut(grid["theta"], bins=30)
    grid["d_elle_bin"] = pd.cut(grid["d_elle"], bins=30)
    heatmap_data = grid.groupby(["d_elle_bin", "theta_bin"]).size().unstack(fill_value=0)

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, cmap="viridis", cbar_kws={'label': 'Conteggio'})
    plt.title("Heatmap di densità congiunta (d_elle vs θ)")
    plt.xlabel("θ (binned)")
    plt.ylabel("d_elle (binned)")
    plt.tight_layout()
    plt.savefig("heatmap_congiunta.png")
    plt.close()

    print("Analisi completata. Grafici salvati nella cartella corrente.")

if __name__ == "__main__":
    analisi_posterior_pmc("posterior_samples_pmc2.csv", theta_true=2423.8)
