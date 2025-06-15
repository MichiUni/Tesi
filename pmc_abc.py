import numpy as np
import torch
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from time import time
from viModel import BayesianEmulator

# 1️⃣ Parametri osservati
freq_obs = np.array([32.63, 96.73, 208.61])  # Frequenze osservate in Hz
sigma_noise = 0.4  # Deviazione standard del rumore misurativo

# 2️⃣ Carica la tua rete bayesiana


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MyNN = BayesianEmulator().to(device)
MyNN.load_state_dict(torch.load("modello_addestrato.pt", map_location=device))
MyNN.eval()

# 3️⃣ Funzione che usa la tua rete (campionando dai pesi ad ogni chiamata)
def forward_network(theta, d_elle, stochastic=True):
    tens = torch.FloatTensor([d_elle, theta]).unsqueeze(0).to(device)
    with torch.no_grad():
        output = MyNN(tens, stochastic=stochastic).cpu().numpy().flatten()
    return output

# 4️⃣ Funzione distanza standardizzata (come nel paper)
def simulate_dist(obs, theta_delle_batch):
    dist = np.zeros(theta_delle_batch.shape[0])
    for i, (theta, d_elle) in enumerate(theta_delle_batch):
        freq = forward_network(theta, d_elle, stochastic=True)
        freq += np.random.normal(0.0, sigma_noise, size=3)
        standardized_diff = (freq - obs) / sigma_noise
        dist[i] = np.linalg.norm(standardized_diff)
    return dist

# 5️⃣ Densità di proposta per PMC
def proposal_density(theta, last_theta, last_w, cov_matrix):
    n_new = theta.shape[0]
    n_old = last_theta.shape[0]
    var = multivariate_normal(mean=np.zeros(2), cov=cov_matrix)
    d = [[last_w[j] * var.pdf(theta[i, :] - last_theta[j, :])
          for i in range(n_new)] for j in range(n_old)]
    d = np.array(d)
    return np.sum(d, axis=0)

# 6️⃣ Campionamento con ABC
def sample_abc_posterior(n, eps, obs, batch_size=1000, use_prior=True, last_theta=None, last_w=None):
    sims_count = 0
    if not use_prior:
        cov_matrix = 2. * np.cov(last_theta, rowvar=False, aweights=last_w)

    theta_accepted = np.zeros((n, 2))
    dist_accepted = np.zeros(n)
    naccepted = 0
    while naccepted < n:
        if use_prior:
            d_elle_batch = np.random.exponential(1.5, batch_size) * 100
            theta_batch = np.random.uniform(0, 5000, batch_size)
            theta_delle = np.column_stack((theta_batch, d_elle_batch))
        else:
            ancestors = np.random.choice(n, size=batch_size, replace=True, p=last_w)
            theta_delle = last_theta[ancestors, :]
            theta_delle += np.random.multivariate_normal(mean=np.zeros(2), cov=cov_matrix, size=batch_size)

        dist = simulate_dist(obs, theta_delle)
        sims_count += batch_size
        w = (dist <= eps) & (theta_delle[:, 0] > 0) & (theta_delle[:, 0] < 5000) & (theta_delle[:, 1] > 0)
        nacc = np.sum(w)
        if nacc == 0:
            continue
        theta_delle = theta_delle[w, :]
        dist = dist[w]
        toadd = min(nacc, n - naccepted)
        theta_accepted[naccepted:naccepted + toadd, :] = theta_delle[:toadd, :]
        dist_accepted[naccepted:naccepted + toadd] = dist[:toadd]
        naccepted += toadd

    if use_prior:
        w = np.ones(n)
    else:
        w = np.exp(-theta_accepted[:, 1] / 150) / proposal_density(theta_accepted, last_theta, last_w, cov_matrix)
    w /= np.sum(w)

    return theta_accepted, dist_accepted, w, sims_count

# 7️⃣ Algoritmo PMC-ABC
def abc_pmc(obs, n_to_accept, prob_fraction, batch_size=1000, max_mins=20):
    start_time = time()
    iteration = 1
    eps = np.inf
    first_iteration = True
    total_sims = 0
    theta = None
    w = None
    history = []

    while (time() - start_time) / 60. < max_mins:
        print(f"Iteration {iteration}, epsilon {eps:.4f}")
        theta, dist, w, sims = sample_abc_posterior(n_to_accept, eps, obs,
                                                     batch_size=batch_size,
                                                     use_prior=first_iteration,
                                                     last_theta=theta,
                                                     last_w=w)
        total_sims += sims
        ess = (np.sum(w) ** 2) / np.sum(w ** 2)
        elapsed_mins = (time() - start_time) / 60.
        print(f"ESS {ess:.0f}, simulations {total_sims}, progress {100. * elapsed_mins / max_mins:.1f}%, minutes {elapsed_mins:.1f}")

        eps = np.quantile(dist, prob_fraction)
        iteration += 1
        first_iteration = False

        # Visualizza istogrammi
        ancestors = np.random.choice(theta.shape[0], size=200, replace=True, p=w)
        df = pd.DataFrame(theta[ancestors, :], columns=['theta', 'd_elle'])
        plt.close()
        f, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(df['theta'], kde=False, binwidth=50, ax=axes[0])
        sns.histplot(df['d_elle'], kde=False, binwidth=10, ax=axes[1])
        f.suptitle(f'Iteration {iteration-1} - epsilon {eps:.4f}')
        f.tight_layout()
        plt.pause(0.5)

        history.append([iteration, eps, elapsed_mins, ess])

    return df, history

# 8️⃣ LANCIO
if __name__ == "__main__":
    df_final, history = abc_pmc(freq_obs, n_to_accept=500, prob_fraction=0.5, batch_size=10000, max_mins=5)
    plt.plot(np.array(history)[:, 1], color='r', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('epsilon')
    plt.title('Evoluzione di epsilon')
    plt.show()
    df_final.to_csv("posterior_samples_pmc2.csv", index=False)
    print("Campioni finali salvati in posterior_samples_pmc2.csv")
