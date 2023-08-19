import torch
from model import DecoderOnlyTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

vocab_size = 10
ndigit = 6
model = DecoderOnlyTransformer(
            num_layers=2,
            num_heads=4,
            hidden_size=128,
            vocab_size=vocab_size,
            block_size=ndigit
        ).to(device)

model.load_state_dict(torch.load("models/reverse_digits.pt"))

x = torch.randint(vocab_size, size=(ndigit,), dtype=torch.long).unsqueeze(0).to(device)
out = model(x)

print("Actual:", torch.flip(x, (-1,)))
print("Predicted:", out.argmax(dim=-1))

# The first half of predictions are wrong, but the second half are correct.
# Because only when the model was predicting for the second half of the sequence, had it already seen the first part of the sequence which are now the labels.