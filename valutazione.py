import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

y_pred = np.loadtxt("predicted_means.txt", delimiter=",")
y_true = np.loadtxt("test_set.txt", delimiter=",")[:, 2:5]

for i, name in enumerate(["f1", "f2", "f3"]):
    y_true_i = y_true[:, i]
    y_pred_i = y_pred[:, i]
    r2 = r2_score(y_true_i, y_pred_i)
    mse = mean_squared_error(y_true_i, y_pred_i)
    rmse = np.sqrt(mse)
    mean_y = np.mean(y_true_i)
    rmse_norm = rmse / mean_y
    print(f"{name.upper()} → R²: {r2:.4f}, MSE: {mse:.4f}, Normalized RMSE: {rmse_norm:.4f} ({rmse_norm*100:.2f}%)")
