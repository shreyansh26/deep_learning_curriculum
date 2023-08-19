import torch
import torch.nn.functional as F
from torch import optim
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
from tqdm import tqdm
from model import DecoderOnlyTransformer

class ReverseDataset(Dataset):
    def __init__(self, ndigit):
        self.ndigit = ndigit
        self.vocab_size = 10 # 10 possible digits 0..9
        self.size = 10**self.ndigit # total number of possible combinations

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.randint(self.vocab_size, size=(self.ndigit,), dtype=torch.long)
        y = torch.flip(x,(-1,)) # Reverse
        return x, y
    
# create a dataset for e.g. 6-digit sequence reversals
ndigit = 6
train_dataset = ReverseDataset(ndigit=ndigit)
print(train_dataset[0])

# Create dataset and dataloader
batch_size = 2048
train_loader = DataLoader(
    train_dataset, shuffle=True, pin_memory=True, batch_size=batch_size
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

model = DecoderOnlyTransformer(
            num_layers=2,
            num_heads=4,
            hidden_size=128,
            vocab_size=train_dataset.vocab_size,
            block_size=train_dataset.ndigit
        ).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

max_epochs = 1
for epoch in range(max_epochs):
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for it, (x, y) in pbar:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        
        logits = model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()

        optimizer.step()
        pbar.set_description(f"epoch {epoch} iter {it}: train loss {loss.item():.5f}")

torch.save(model.state_dict(), "models/reverse_digits.pt")