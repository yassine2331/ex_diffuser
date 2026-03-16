
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from configs.diffusion_config import TrainingConfig,AttentionConfig ,celebA



from data.dataloader import get_dataloader,get_dataloader_test

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Hyperparameters
LR = 0.001
BATCH_SIZE = 64
EPOCHS = 1

# Transforms (padding to get 32x32 input)
transform = transforms.Compose([
    transforms.Pad(2),
    transforms.ToTensor()
])

# MNIST Datasets
#train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transform, download=True)
#test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transform, download=True)

# Data Loaders
#train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
#test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
config = celebA()
train_loader = get_dataloader(config,concept=config.concept)
test_loader = get_dataloader_test(config,concept=config.concept)

# Initialize model, loss, and optimizer
model = DiffusersLatentClassifier(input_channels=16, num_classes=40).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Accuracy evaluation (used only for test set)
def evaluate_accuracy(loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for step , batch in enumerate(loader):
            data = batch["images"]
            targets = batch.get("concepts", None) 
            data = data.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(data)
            _, predictions = outputs.max(1)
            correct += (predictions == torch.argmax(targets,dim=1)).sum().item()
            total += targets.size(0)

    model.train()
    return 100.0 * correct / total

best_accuracy = 0.0

running_losses=[]
train_accies = []
test_accies = []
# Training Loop
for epoch in range(EPOCHS):
    
    print("EPOCH", epoch)
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for step , batch in enumerate(train_loader):
        
        data = batch["images"]
        targets = batch.get("concepts", None) 

        data = data.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Forward
        outputs = model(data)
        loss = criterion(outputs, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stats
        running_loss += loss.item()
        _, predictions = outputs.max(1)
        correct += (predictions == torch.argmax(targets,dim=1)).sum().item()
        total += targets.size(0)

    train_acc = 100.0 * correct / total
    test_acc = evaluate_accuracy(test_loader, model)
    
    running_losses.append(running_loss)
    train_accies.append(train_acc)
    test_accies.append(test_acc)

    print(f"Epoch [{epoch+1}/{EPOCHS}], "
          f"Loss: {running_loss:.4f}, "
          f"Train Acc: {train_acc:.2f}%, "
          f"Test Acc: {test_acc:.2f}%")
    # Save the best model
    if test_acc > best_accuracy:
        best_accuracy = test_acc
        torch.save(model.state_dict(), "dummy_model.pth")
        print("Best model saved with accuracy:", best_accuracy)

# Plotting
