import torch
import numpy as np
import matplotlib.pyplot as plt
from viModel import BayesianEmulator, StandardEmulator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bayesian_model = BayesianEmulator().to(device)
standard_model = StandardEmulator().to(device)

# Carica i parametri salvati
bayesian_model.load_state_dict(torch.load("modello_addestrato.pt", map_location=device))
standard_model.load_state_dict(torch.load("modello_standard.pt", map_location=device))

# Funzione per ottenere i parametri
def get_bayesian_params(layer):
    mean = layer.weights_mean.detach().cpu().numpy().flatten()
    sigma = torch.exp(layer.lweights_sigma).detach().cpu().numpy().flatten()
    if layer.has_bias:
        bias_mean = layer.bias_mean.detach().cpu().numpy().flatten()
        bias_sigma = torch.exp(layer.lbias_sigma).detach().cpu().numpy().flatten()
    else:
        bias_mean, bias_sigma = None, None
    return mean, sigma, bias_mean, bias_sigma

def get_standard_params(layer):
    weight = layer.weight.detach().cpu().numpy().flatten()
    if layer.bias is not None:
        bias = layer.bias.detach().cpu().numpy().flatten()
    else:
        bias = None
    return weight, bias

bayesian_layers = [bayesian_model.hidden, bayesian_model.output]
standard_layers = [standard_model.hidden, standard_model.output]

# Crea i plot
for idx, (bayes_layer, std_layer) in enumerate(zip(bayesian_layers, standard_layers)):
    bayes_weight_mean, bayes_weight_sigma, bayes_bias_mean, bayes_bias_sigma = get_bayesian_params(bayes_layer)
    std_weight, std_bias = get_standard_params(std_layer)

    # Plot pesi
    for i, (mean, sigma, std_val) in enumerate(zip(bayes_weight_mean, bayes_weight_sigma, std_weight)):
        x = np.linspace(mean - 4*sigma, mean + 4*sigma, 1000)
        y = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mean) / sigma)**2)

        plt.figure()
        plt.plot(x, y, label="Bayesian weight", color='blue')
        plt.axvline(std_val, color='red', linestyle='--', label="Standard weight")
        plt.title(f"Layer {idx+1} - Weight {i}")
        plt.xlabel("Weight value")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Plot bias
    if bayes_bias_mean is not None and std_bias is not None:
        for i, (mean, sigma, std_val) in enumerate(zip(bayes_bias_mean, bayes_bias_sigma, std_bias)):
            x = np.linspace(mean - 4*sigma, mean + 4*sigma, 1000)
            y = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mean) / sigma)**2)

            plt.figure()
            plt.plot(x, y, label="Bayesian bias", color='blue')
            plt.axvline(std_val, color='red', linestyle='--', label="Standard bias")
            plt.title(f"Layer {idx+1} - Bias {i}")
            plt.xlabel("Bias value")
            plt.ylabel("Density")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

print("Plot completati!")
