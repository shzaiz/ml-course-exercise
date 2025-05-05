import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import AlexNet

transform = transforms.Compose([
    transforms.Resize(227),  # AlexNet expects 227x227 input
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

train_dataset = torchvision.datasets.MNIST(root='../ignore', train=True, download=True, transform=transform)
#                                          ^~~~~~~~~~~~~~~~ Avoiding commiting big files
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = torchvision.datasets.MNIST(root='../ignore', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

def train(mod, device, train_loader, optimizer, criterion, epoch):
    mod.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = mod(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]  Loss: {loss.item():.6f}')

def test(mod, device, test_loader, criterion):
    mod.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = mod(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)\n')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mod = AlexNet(10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(mod.parameters(), lr=0.01, momentum=0.9)

num_epochs = 10
for epoch in range(1, num_epochs + 1):
    train(mod, device, train_loader, optimizer, criterion, epoch)
    test(mod, device, test_loader, criterion)

torch.save(mod.state_dict(), 'alexnet_mnist.pth')