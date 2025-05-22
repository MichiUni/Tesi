import torch
import numpy as np
from viModel import BayesianEmulator
from salva_posterior import salva_posterior

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carica modello
model = BayesianEmulator()
model.load_state_dict(torch.load("modello_addestrato.pt", map_location=device))
model.to(device)
model.eval()

# Costanti di scala (non normalizzate)
#X_MAX = [2639.7, 4999.9]
#Y_MAX = [43.433, 121.31, 265.22]
y_obs = torch.tensor([32.6263, 96.7327, 208.6077], dtype=torch.float32).to(device)

def run_abc(S=1000, n_mc=100, epsilon=25.0, output_file="posterior_bayesiano.csv"):
    posterior = []

    for s in range(S):
        x = np.random.exponential(1.5) * 100
        theta = np.random.uniform(0, 5000)
        inputs = torch.tensor([[x, theta]], dtype=torch.float32).to(device)
        acc = 0

        with torch.no_grad():
            for _ in range(n_mc):
                output = model(inputs, stochastic=True)[0]
                dist = torch.norm(output - y_obs)
                if dist.item() <= epsilon:
                    acc += 1

        posterior.append([x, theta, acc])
        print(f"Simulazione {s+1}/{S} → accettazioni: {acc}")

    salva_posterior(posterior, filename=output_file)
    print(f"ABC completato. Risultati salvati in '{output_file}'.")

#if __name__ == "__main__":
#    run_abc(S=1000, n_mc=100, epsilon=25, output_file="posterior_bayesiano.csv")

#if __name__ == "__main__":
#    run_abc(S=5000, n_mc=100, epsilon=20, output_file="posterior_eps20.csv")

if __name__ == "__main__":
    run_abc(S=30000, n_mc=100, epsilon=2, output_file="posterior_eps10.csv")
