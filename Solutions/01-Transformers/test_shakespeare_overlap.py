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
        return self.words_count - self.block_size

    def __getitem__(self, idx):
        # Get chunks (non-overlapping from previous chunk)
        chunk = self.words[idx:idx + self.block_size + 1]
        # Encode to idx
        idxs = [self.word_to_idx[w] for w in chunk]

        x = torch.tensor(idxs[:-1], dtype=torch.long)
        y = torch.tensor(idxs[1:], dtype=torch.long)
        return x, y

train_dataset = ShakespeareDataset(text, block_size)
print(train_dataset.vocab_size)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print("device:", device)

model = DecoderOnlyTransformer(
            num_layers=8,
            num_heads=8,
            hidden_size=512,
            vocab_size=train_dataset.vocab_size,
            block_size=train_dataset.block_size
        ).to(device)

model.load_state_dict(torch.load("models/shakespeare_works_overalap.pt"))

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

context = " O god, O god! "
x = torch.tensor([train_dataset.word_to_idx[s] for s in re.split(r"\b", context)], dtype=torch.long).unsqueeze(0).to(device)
y = sample(model, x, 500, temperature=1.0, sample=True, top_k=10)[0]
completion = ''.join([train_dataset.idx_to_word[int(i)] for i in y])
print(completion)

'''
Prompt - " O god, O god! "
Completion -
 O god, O god! O most foul, bear patience!
O bear your murd’er-night, which you seem’d to tremble,
As seemeth as looks in nature, do note you now,
Must leave your pointing and us in our labour’s
To shine too near us, and at all things shall be peace.

PUCK.
My fairy lord, this must be done with haste,
For night’s swift dragons cut the clouds full fast;
And yonder shines Aurora’s harbinger,
At whose approach, ghosts wandering here and there
Troop home to churchyards. Damnèd spirits all,
That in cross-ways and floods adore,
In mere simplicity she joined a league,
And humbly sues, and with a Trumpet.

ULYSSES.
Why thus began you further?

TROILUS.
No, by that good time to that chances our person
And by our holy garments rejoice in authority
With this detested discourse.
Why should the labouring come unto our master?
O holy-lived fair court or painted evil,
On pain to be ingrate.

AUDREY.
I prithee let me hear no more.

Enter a Messenger.

MESSENGER.
Well said, my noble Scot. Good Griffith,
Those looks are the cause of heaven. Bring me them now
To this hungry age at this heart.
Two such tricks of mine can yield me no more;
Since the love that makes us breath do lie.

CASSIUS.
They say that they know by words of Pompey the whilst your lordship shall be
delivered.

BRUTUS.
Why, adieu.

CORIOLANUS.
A sight I thought the rather trembles in his drowsy age. It
'''

'''
Prompt - " O dear! "
Completion -
O dear! I
knew by birth of my love; and let that grieve thine eye prove,
For it hath been bitter very cold drops away.
Go in scarlet about the children of this world,
My noble lords, and hear of your son.

KING.
Do not you love him, nor him any time he keeps
To ruminate it bravely, and perform’d my will
Unto this duke’s ear: he swore, some —for him I swear and true,
And I have banished him from his company
A single hand would have some speech.

ADRIANA.
What did you swear before?

LUCIANA.
I swore the fault unto a thing consent
Within an honest face. Alack the season lies.
Where is the sick and ugly man?

ADRIANA.
He came unto the hearty welcome.

tide’ excel of it! And sure as we pass along,
And he that dies when we do determine from hence.

EXETER.
There is some peevish brat hath miscarried. One grief
Is oft alone; the phoenix’ throne himself,
As wakes to the west!

IAGO.
If she do, you will marry her upon her,
As she do the rarest gambols’ love, when they think.

OTHELLO.
O, before this lady knows
That she is still the woman and all her sons,
And if she should be miss’d to know
Her estimation behalf, her greatness, and the worth
Was all the means that have miscarried
By image their own beauties: or, if you have not,
An if you do, to perform it in your age,
A
'''