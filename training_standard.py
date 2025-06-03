import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from viModel import StandardEmulator

# -------------------- Dataset --------------------
class FrequencyDataset(Dataset):
    def __init__(self, filepath):
        data = torch.tensor(np.loadtxt(filepath, delimiter=","), dtype=torch.float32)
        self.x = data[:, :2]
        self.y = data[:, 2:5]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# -------------------- Training --------------------
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset e dataloader
    train_dataset = FrequencyDataset("training_set.txt")
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    model = StandardEmulator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(10000):
        model.train()
        total_loss = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            preds = model(x_batch)
            loss = criterion(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # Salva il modello
    torch.save(model.state_dict(), "modello_standard.pt")
    print("Modello standard salvato in 'modello_standard.pt'")

if __name__ == "__main__":
    train_model()
