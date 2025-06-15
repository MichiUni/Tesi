import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error
from viModel import StandardEmulator
import time

# 2. Carica il modello e i pesi salvati
model = StandardEmulator()
model.load_state_dict(torch.load("modello_standard.pt"))
model.eval()

# 3. Carica il test set
X_test = np.loadtxt("test_set.txt", delimiter=",", usecols=(0, 1))
y_test = np.loadtxt("test_set.txt", delimiter=",", usecols=(2, 3, 4))

X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

# 4. Ottieni le predizioni
with torch.no_grad():
    start_total=time.time()
    y_pred = model(X_test).numpy()
    end_total=time.time()

n_test_samples=len(X_test)
print(f"Tempo medio per predizione: {(end_total-start_total)/n_test_samples:.10f} secondi")
print(f"Tempo totale inferenza test set: {end_total-start_total:.10f} secondi")

# 5. Calcola R², MSE e RMSE normalizzato
for i, name in enumerate(["f1", "f2", "f3"]):
    y_true_i = y_test[:, i].numpy()
    y_pred_i = y_pred[:, i]

    r2 = r2_score(y_true_i, y_pred_i)
    mse = mean_squared_error(y_true_i, y_pred_i)
    rmse = np.sqrt(mse)
    mean_y = np.mean(y_true_i)
    rmse_norm = rmse / mean_y

    print(f"{name.upper()} → R²: {r2:.4f}, MSE: {mse:.4f}, Normalized RMSE: {rmse_norm:.4f} ({rmse_norm*100:.2f}%)")
