from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
import random

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
        return x

class PairwiseDataset(Dataset):
    def __init__(self):
        super().__init__()
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        self.orig_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

        self.curr_dataset, self.holdout_dataset = torch.utils.data.random_split(self.orig_dataset, [55000, 5000])

    def __len__(self):
        return len(self.curr_dataset)

    def __getitem__(self, idx):
        data1, label1 = self.curr_dataset[idx]
        same_class = random.randint(0, 1)
        
        if same_class:
            while 1:
                data2, label2 = random.choice(self.curr_dataset)
                if label1 == label2 and not torch.all(torch.eq(data1, data2)):
                    break
        else:
            while 1:
                data2, label2 = random.choice(self.curr_dataset)
                if label1 != label2:
                    break

        label = torch.tensor([label1 == label2], dtype=torch.float32)
        return (data1, data2, label)

def contrastive_loss_fn(output0, output1, target, epsilon=1.5):
    # Standard Contrastive loss - https://lilianweng.github.io/posts/2021-05-31-contrastive/
    # Epsilon/Margin = 1.5
    dist = torch.nn.functional.pairwise_distance(output0, output1, keepdim=True)
    loss = (target * dist**2) + ((1 - target) * torch.clamp(epsilon - dist, min=0.0)**2)
    loss = loss.mean()
    return loss

def train(args, model, device, train_loader, optimizer, loss_fn, epoch):
    model.train()
    for batch_idx, (data0, data1, target) in enumerate(train_loader):
        data0, data1, target = data0.to(device), data1.to(device), target.to(device)

        optimizer.zero_grad()

        output0 = model(data0)
        output1 = model(data1)

        loss = loss_fn(output0, output1, target)

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data0), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader, holdout_samples):
    model.eval()
    test_accuracy = 0
    
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            output = model(data)
            pred = None
            min_score = float('inf')
            for holdout_data, label in holdout_samples:
                output_to_match = model(holdout_data)
                score = torch.nn.functional.pairwise_distance(output, output_to_match).item()

                if score < min_score:
                    pred = label
                    min_score = score

            test_accuracy += (pred == target).item()

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_accuracy, len(test_loader.dataset),
        (100. * test_accuracy / len(test_loader.dataset))))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='For Saving the final Model from all ranks after training')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': 1}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': False}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    dataset_train = PairwiseDataset()

    dataset_test = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    model = Net().to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    loss_fn = contrastive_loss_fn
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    holdout_sample = []
    for i in range(10):
        for data, label in dataset_train.holdout_dataset:
            if label == i:
                holdout_sample.append((data.unsqueeze(0).to(device), label))
                break

    assert len(holdout_sample) == 10

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, loss_fn, epoch)
        test(model, device, test_loader, holdout_sample)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()