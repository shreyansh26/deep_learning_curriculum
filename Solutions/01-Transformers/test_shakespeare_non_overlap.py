import re
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from model import DecoderOnlyTransformer

block_size = 128
text = open('data/100-0.txt', 'r', encoding="utf8").read()

class ShakespeareDataset(Dataset):
    def __init__(self, text, block_size):
        self.words = re.split(r"\b", text)
        self.vocab = sorted(list(set(self.words)))
        self.block_size = block_size
        self.words_count = len(self.words)
        self.vocab_size = len(self.vocab)

        self.word_to_idx = {word: i for i, word in enumerate(self.vocab)}
        self.idx_to_word = {i: word for i, word in enumerate(self.vocab)}

    def __len__(self):
        return ((self.words_count) // self.block_size)

    def __getitem__(self, idx):
        # Get chunks (non-overlapping from previous chunk)
        chunk = self.words[idx * self.block_size: (idx + 1) * self.block_size]
        # Encode to idx
        idxs = [self.word_to_idx[w] for w in chunk]

        x = torch.tensor(idxs[:-1], dtype=torch.long)
        y = torch.tensor(idxs[1:], dtype=torch.long)
        return x, y
    
train_dataset = ShakespeareDataset(text, block_size)
print(train_dataset.vocab_size)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("device:", device)

model = DecoderOnlyTransformer(
            num_layers=8,
            num_heads=8,
            hidden_size=512,
            vocab_size=train_dataset.vocab_size,
            block_size=train_dataset.block_size
        ).to(device)

model.load_state_dict(torch.load("models/shakespeare_works_non_overalap.pt"))

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time
    """
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        logits = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x

context = ''' Julius Caesar was a famous man. '''

inp = torch.tensor([train_dataset.word_to_idx[s] for s in re.split(r"\b", context)], dtype=torch.long).unsqueeze(0).to(device)
y = sample(model, inp, 300, temperature=1.0, sample=True, top_k=10)[0]
completion = ''.join([train_dataset.idx_to_word[int(i)] for i in y])
print("Completion:", completion)

'''
Julius Caesar was a famous man. His brother was but
mad.

SICINIUS.
In the next king’s crown, sir, were as his great rewards fit the state
of all the love of Egypt.

BRUTUS.
Well, we doubt not but by th’ advice, no: all this faults was wont to
have been heard again in the great men their company, for their faces we have a a a
fancy, or note which a fancy, as a is will not a a a married a almost
AMIENS.

GENTLEWOMANmuch wager’S-what the for word a a man is man proves; here must playing comes
Until a last, or must a throstle man censure must a must must make make will cry be a
Something? And married with them of Don: would Don an or with almost note almost last would
them?

MERCHANT.
No would a rather note would would, sir, were cry, “you heard at heard bold heard
'''