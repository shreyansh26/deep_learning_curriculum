import re
import torch
from torch import optim
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from model import DecoderOnlyTransformer

class ShaekspeareDataset(Dataset):
    def __init__(self, text, block_size):
        self.words = re.split(r"\b", text)
        self.vocab = sorted(list(set(self.words)))
        self.block_size = block_size
        self.words_count = len(self.words)
        self.vocab_size = len(self.vocab)

        self.word_to_idx = {word: i for i, word in enumerate(self.vocab)}
        self.idx_to_word = {i: word for i, word in enumerate(self.vocab)}

    def __len__(self):
        return self.words_count - self.block_size

    def __getitem__(self, idx):
        # Get chunks (non-overlapping from previous chunk)
        chunk = self.words[idx:idx + self.block_size + 1]
        # Encode to idx
        idxs = [self.word_to_idx[w] for w in chunk]

        x = torch.tensor(idxs[:-1], dtype=torch.long)
        y = torch.tensor(idxs[1:], dtype=torch.long)
        return x, y
    
# create a dataset for e.g. 6-digit sequence reversals
block_size = 128
text = open('data/100-0.txt', 'r').read()
train_dataset = ShaekspeareDataset(text, block_size)
print(len(train_dataset))

# Create dataset and dataloader
batch_size = 512
train_loader = DataLoader(
    train_dataset, shuffle=True, pin_memory=True, batch_size=batch_size, drop_last=True
)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print("device:", device)

model = DecoderOnlyTransformer(
            num_layers=8,
            num_heads=8,
            hidden_size=512,
            vocab_size=train_dataset.vocab_size,
            block_size=train_dataset.block_size
        ).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4)

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

torch.save(model.state_dict(), "models/shakespeare_works_overalap.pt")