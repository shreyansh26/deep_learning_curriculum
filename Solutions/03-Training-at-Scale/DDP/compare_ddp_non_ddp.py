import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def load_model(prefix="initial", rank=None):
    model = Net()
    if rank is not None:
        model.load_state_dict(torch.load(f"models/{prefix}_mnist_cnn_rank_{rank}.pt"))
    else:
        model.load_state_dict(torch.load(f"models/{prefix}_mnist_cnn.pt"))
    model = model.to("cuda:0")
    return model

def verify_same_model(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True

model_no_dp_initial = load_model("initial", None)
model_dp_initial_0 = load_model("initial", 0)
model_dp_initial_1 = load_model("initial", 1)

print(verify_same_model(model_dp_initial_0, model_dp_initial_1))
print(verify_same_model(model_dp_initial_0, model_no_dp_initial))

# Final models won't be same for normal and DDP implementations as we do DDP and not DP.
# Difference - https://discuss.pytorch.org/t/is-data-parallel-or-ddp-equivalent-to-larger-batch/104825/6