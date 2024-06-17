import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets, models
import os
from tqdm import tqdm
from torch_lr_finder import LRFinder
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([
    transforms.Resize((128, 128)),  
    transforms.ToTensor(),          
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
])

class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)
    
    def find_classes(self, directory):
        classes = [d.name for d in os.scandir(directory) if d.is_dir() and not d.name.startswith('.')]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


data_dir = '/kaggle/input/datahands/content/hand_cropped'
dataset = CustomImageFolder(root=data_dir, transform=transform)


train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


model = models.resnet18(pretrained=True)


num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 6)


model = torch.nn.DataParallel(model)
model.cuda()


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
lr_finder.range_test(train_loader, end_lr=10, num_iter=100)
lr_finder.plot()  
lr_finder.reset()  


lr = lr_finder.history["lr"][lr_finder.history["loss"].index(lr_finder.best_loss)]


def calculate_accuracy(loader, model):
    correct = 0
    total = 0
    model.eval()  
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.cuda(), labels.cuda()  # Use GPU
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()  
    return correct / total


num_epochs = 50
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
early_stopping = False
min_loss = np.Inf
best_epoch = 0
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    if early_stopping:
        break
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch'):
        inputs, labels = inputs.cuda(), labels.cuda()  
        
        optimizer.zero_grad()

        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    train_acc = correct / total
    test_acc = calculate_accuracy(test_loader, model)
    train_losses.append(epoch_loss)
    train_accuracies.append(train_acc)
    test_losses.append(epoch_loss)
    test_accuracies.append(test_acc)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')
    
    if epoch_loss < min_loss:
        min_loss = epoch_loss
        best_epoch = epoch
        torch.save(model.state_dict(), 'hand_gesture_cnn_best.pth')
    else:
        if epoch - best_epoch > 5:  
            print("Early stop")
            early_stopping = True

    scheduler.step()  


torch.save(model.state_dict(), 'hand_gesture_cnn_final_res18.pth')
print(f'Best epoch: {best_epoch+1}')
print(f'Best loss: {min_loss:.4f}')
print(f'Best train accuracy: {train_accuracies[best_epoch]:.4f}')
print(f'Best test accuracy: {test_accuracies[best_epoch]:.4f}')
epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'b', label='Training loss')
plt.plot(epochs, test_losses, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, 'b', label='Training accuracy')
plt.plot(epochs, test_accuracies, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
