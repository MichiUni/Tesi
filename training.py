from viModel import BayesianEmulator
from dataset import FrequencyDataset
from torch.utils.data import DataLoader
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = FrequencyDataset("training_set.txt")
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

model = BayesianEmulator().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10000):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        preds = model(x, stochastic=True)
        mse_loss = torch.nn.functional.mse_loss(preds, y)
        kl_loss = model.evalAllLosses()
        loss = mse_loss + kl_loss / len(train_dataset)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "modello_addestrato.pt")
