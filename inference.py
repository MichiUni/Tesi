import torch
import numpy as np
from torch.utils.data import DataLoader
from viModel import BayesianEmulator
from dataset import FrequencyDataset
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BayesianEmulator()
model.load_state_dict(torch.load("modello_addestrato.pt", map_location=device))
model.to(device)
model.eval()

test_dataset = FrequencyDataset("test_set.txt")
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

n_samples = 100
all_means = []
all_stds = []

start_total = time.time()
with torch.no_grad():
    for x_batch, _ in test_loader:
        x_batch = x_batch.to(device)
        mc_outputs = torch.stack([model(x_batch, stochastic=True) for _ in range(n_samples)])
        batch_mean = mc_outputs.mean(dim=0)
        batch_std = mc_outputs.std(dim=0)
        all_means.append(batch_mean.cpu())
        all_stds.append(batch_std.cpu())

end_total = time.time()
n_test_samples = len(test_dataset)
print(f"Tempo medio per predizione: {(end_total - start_total) / n_test_samples:.6f} secondi")
print(f"Tempo totale inferenza test set: {end_total - start_total:.6f} secondi")

all_means = torch.cat(all_means, dim=0).numpy()
all_stds = torch.cat(all_stds, dim=0).numpy()

np.savetxt("predicted_means.txt", all_means, delimiter=",")
np.savetxt("predicted_stds.txt", all_stds, delimiter=",")
