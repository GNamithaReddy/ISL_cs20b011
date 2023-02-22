import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor
import numpy as np

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

new_labels = {
    0: 'Upper',
    1: 'Upper',
    2: 'Upper',
    3: 'Upper',
    4: 'Lower',
    5: 'Lower',
    6: 'Feet',
    7: 'Feet',
    8: 'Feet',
    9: 'Bag'
}
new_train_data = []
new_train_targets = []

for i, (data, target) in enumerate(training_data):
    if target in [0, 1, 2, 3]:  # Upper
        new_target = 0
    elif target in [4, 5]:  # Lower
        new_target = 1
    elif target in [6, 7, 8]:  # Feet
        new_target = 2
    else:  # Bag
        new_target = 3
    new_train_data.append(data)
    new_train_targets.append(new_target)
    
    new_test_data = []
new_test_targets = []
for i, (data, target) in enumerate(test_data):
    if target in [0, 1, 2, 3]:  # Upper
        new_target = 0
    elif target in [4, 5]:  # Lower
        new_target = 1
    elif target in [6, 7, 8]:  # Feet
        new_target = 2
    else:  # Bag
        new_target = 3
    new_test_data.append(data)
    new_test_targets.append(new_target)
    
    class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)
      
custom_trainset = CustomDataset(new_train_data, new_train_targets)
custom_testset = CustomDataset(new_test_data, new_test_targets)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )
        

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
       

model = Net().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# Define the batch size
batch_size = 64

# Create DataLoaders for the train and test sets
trainloader = torch.utils.data.DataLoader(custom_trainset, batch_size=batch_size,shuffle=True)
testloader = torch.utils.data.DataLoader(custom_testset, batch_size=batch_size,shuffle=False)

def map_labels_to_ints(label):
    if label == 'Lower':
        return 0
    elif label == 'Upper':
        return 1
    else:
        raise ValueError(f"Unknown label: {label}")


for X, y in testloader:
    print(f"X: {X}")
    print(f"y: {y}")
    print(f"Type of X: {type(X)}")
    print(f"Size of X: {X.size}")
    print(f"Type of y: {type(y)}")
    break

for X,y in testloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

import torch
import torch.nn as nn
import torch.optim as optim

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")  

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(trainloader,model, loss_fn, optimizer)
    test(testloader,model,loss_fn)
print("Done!")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

model = Net()
model.load_state_dict(torch.load("model.pth"))

classes = [
    "Upper",
    "Lower",
    "Feet",
    "Bag"
]

model.eval()
x, y = custom_testset[0][0], custom_testset[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')




