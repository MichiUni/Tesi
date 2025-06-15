import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def analisi_posterior_standard(file_csv="posterior_standard_top0_5perc_db.csv", theta_true=2423.8):
    # Caricamento dati
    df = pd.read_csv(file_csv)

    # ========================
    # 1. Posterior distribution di θ
    # ========================
    plt.figure(figsize=(10, 5))
    sns.histplot(df["theta"], bins=100, kde=True, color="dodgerblue", edgecolor="black")
    plt.axvline(theta_true, color="red", linestyle="--", label="θ true")
    plt.axvline(df["theta"].mean(), color="blue", linestyle="--", label="Media posterior")
    plt.title("Distribuzione a posteriori di θ (location del danno)")
    plt.xlabel("θ (mm)")
    plt.ylabel("Densità")
    plt.legend()
    plt.tight_layout()
    plt.savefig("posterior_theta_bayesiana.png")
    plt.close()

    # ========================
    # 2. Jointplot θ vs x (colorato per distanza)
    # ========================
    jp = sns.jointplot(data=df, x="theta", y="x", hue="distanza", palette="viridis", height=6, alpha=0.7)
    jp.fig.suptitle("Jointplot θ vs x (colorato per distanza)", fontsize=12)
    jp.fig.tight_layout()
    jp.fig.subplots_adjust(top=0.95)
    jp.savefig("jointplot_theta_x_distanza.png")
    plt.close()

    # ========================
    # 3. Zoom su θ ∈ [2000, 3000]
    # ========================
    df_zoom = df[(df["theta"] >= 2000) & (df["theta"] <= 3000)]
    plt.figure(figsize=(8, 5))
    plt.scatter(df_zoom["theta"], df_zoom["distanza"], alpha=0.5, s=10, color="darkgreen")
    plt.axvline(theta_true, color="red", linestyle="--", label="θ true")
    plt.title("Zoom: θ ∈ [2000, 3000]")
    plt.xlabel("θ")
    plt.ylabel("Distanza")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("zoom_theta_2000_3000.png")
    plt.close()

    # ========================
    # 4. Heatmap distanza media (x vs θ)
    # ========================
    grid = df.copy()
    grid["x_bin"] = pd.cut(grid["x"], bins=30)
    grid["theta_bin"] = pd.cut(grid["theta"], bins=30)
    pivot = grid.pivot_table(index="x_bin", columns="theta_bin", values="distanza", aggfunc="mean")

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, cmap="viridis", cbar_kws={'label': 'Distanza media'})
    plt.title("Heatmap della distanza media (x vs θ)")
    plt.xlabel("θ (binned)")
    plt.ylabel("x (binned)")
    plt.tight_layout()
    plt.savefig("heatmap_distanza_media.png")
    plt.close()

    print("Grafici generati e salvati.")

if __name__ == "__main__":
    analisi_posterior_standard("posterior_standard_top0_5perc.csv", theta_true=2423.8)