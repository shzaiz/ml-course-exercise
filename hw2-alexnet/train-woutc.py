import torch.optim as optim
import torch
import torch.nn as nn
import torchvision
from model import MLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Adam 收敛更快 :contentReference[oaicite:6]{index=6}

train_dataset = torchvision.datasets.MNIST(root='../ignore', train=True, download=False, transform=transform)
#                                          ^~~~~~~~~~~~~~~~ Avoiding commiting big files
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = torchvision.datasets.MNIST(root='../ignore', train=False, download=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}')

def test():
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(dim=1)
            correct += (pred == target).sum().item()
    print(f'Test Accuracy: {100.*correct/len(test_ds):.2f}%')

for epoch in range(1, 11):
    train(epoch)
    test()