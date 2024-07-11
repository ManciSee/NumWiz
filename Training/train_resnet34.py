import splitfolders
import os
import json
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt
import random
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import numpy as np
from tqdm import tqdm


# Comment this if you have already splited the dataset
splitfolders.ratio("input_dataset", # The location of dataset
                   output="output_dataset", # The output location
                   seed=42, # The number of seed
                   ratio=(.7, .3), # The ratio of splited dataset
                   group_prefix=None, # If your dataset contains more than one file like ".jpg", ".pdf", etc
                   move=False # If you choose to move, turn this into True
                   )

test_path = "PATH_TO_TEST_DATASET"
train_path = "PATH_TO_TRAIN_DATASET"

def build_json(directory_string, output_json_name):
    directory = directory_string
    class_lst = os.listdir(directory) 
    class_lst.sort() 
    data = []
    for class_name in class_lst:
        class_path = os.path.join(directory, class_name)
        file_list = os.listdir(class_path) 
        for file_name in file_list:
            file_path = os.path.join(directory, class_name, file_name)
            data.append({
                'file_name': file_name,
                'file_path': file_path,
                'class_name': class_name,
                'class_index': class_lst.index(class_name)
            }) 

    with open(output_json_name, 'w') as jsonfile:
        json.dump(data, jsonfile, indent=4)
    return


build_json(train_path, 'train.json')
build_json(test_path, 'test.json')

train_df = pd.read_json('train.json')
test_df = pd.read_json('test.json')

train_df.head()

class_names = list(train_df['class_name'].unique())
#['0', '1', '2', '3', '4', '5']

root_dir = "ROOT_DIR_OF_DATASET"

train_json_path = "PATH_TO_TRAIN_JSON"
test_json_path = "PATH_TO_TEST_JSON"

# Define transforms for train and test datasets
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define dataset class
class HandDataset(Dataset):
    def __init__(self, json_file, root_dir, transform=None):
        self.annotation_df = pd.read_json(json_file)
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(self.annotation_df['class_index'].unique())  
    
    def __len__(self):
        return len(self.annotation_df)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.annotation_df.iloc[idx]['file_path'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        class_index = self.annotation_df.iloc[idx]['class_index']
        if self.transform:
            image = self.transform(image)
        return image, class_index

train_dataset = HandDataset(json_file=train_json_path, root_dir=root_dir, transform=transform_train)
test_dataset = HandDataset(json_file=test_json_path, root_dir=root_dir, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet34(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))  
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

weight_dir = 'FOLDER_TO_SAVE_WEIGHTS'
if not os.path.exists(weight_dir):
    os.makedirs(weight_dir)

def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total_preds += labels.size(0)
        correct_preds += (predicted == labels).sum().item()
        
        pbar.set_postfix({'Loss': loss.item()})
    
    train_loss = running_loss / len(train_loader.dataset)
    train_acc = correct_preds / total_preds
    train_f1 = f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')
    
    return train_loss, train_acc, train_f1

def evaluate(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(test_loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_preds += labels.size(0)
            correct_preds += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'Loss': loss.item()})
    
    test_loss = running_loss / len(test_loader.dataset)
    test_acc = correct_preds / total_preds
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    test_precision = precision_score(all_labels, all_preds, average='macro')
    test_recall = recall_score(all_labels, all_preds, average='macro')
    
    return test_loss, test_acc, test_f1, test_precision, test_recall, all_labels, all_preds


def denormalize(image, mean, std):
    img = image.clone()  
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img


def class_accuracy(preds, labels, num_classes):
    correct = [0] * num_classes
    total = [0] * num_classes
    for i in range(len(labels)):
        label = labels[i]
        pred = preds[i]
        if label == pred:
            correct[label] += 1
        total[label] += 1
    acc = [c / t if t > 0 else 0 for c, t in zip(correct, total)]
    return acc

num_epochs = 10
best_test_acc = 0.0
best_model_file = None

train_losses = []
train_accuracies = []
train_f1_scores = []
test_losses = []
test_accuracies = []
test_f1_scores = []
test_precisions = []
test_recalls = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    
    train_loss, train_acc, train_f1 = train(model, train_loader, criterion, optimizer, epoch)
    test_loss, test_acc, test_f1, test_precision, test_recall, all_labels, all_preds = evaluate(model, test_loader, criterion)
    
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    train_f1_scores.append(train_f1)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    test_f1_scores.append(test_f1)
    test_precisions.append(test_precision)
    test_recalls.append(test_recall)
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")
    print(f"Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}")
    print()
    
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_model_file = os.path.join(weight_dir, f"resnet18_hagrid_best.pth")
        torch.save(model.state_dict(), best_model_file)
        print(f"Saved best model weights with test accuracy {test_acc:.4f} at {best_model_file}")

print(f"Loading best model weights from {best_model_file}")
model.load_state_dict(torch.load(best_model_file))

final_test_loss, final_test_acc, final_test_f1, final_test_precision, final_test_recall, all_labels, all_preds = evaluate(model, test_loader, criterion)

print(f"Final Test Loss: {final_test_loss:.4f}, Final Test Acc: {final_test_acc:.4f}, Final Test F1: {final_test_f1:.4f}")
print(f"Final Test Precision: {final_test_precision:.4f}, Final Test Recall: {final_test_recall:.4f}")

# Plotting losses
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train')
plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.show()

# Plotting accuracies
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train')
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Test')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy')
plt.legend()
plt.show()

# Plotting F1 scores
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_f1_scores) + 1), train_f1_scores, label='Train')
plt.plot(range(1, len(test_f1_scores) + 1), test_f1_scores, label='Test')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.title('Training and Test F1 Score')
plt.legend()
plt.show()

# Plotting Precision and Recall
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(test_precisions) + 1), test_precisions, label='Precision')
plt.plot(range(1, len(test_recalls) + 1), test_recalls, label='Recall')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.title('Test Precision and Recall')
plt.legend()
plt.show()

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Displaying test images with predicted classes
model.eval()
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
fig, axes = plt.subplots(2, 5, figsize=(20, 8))

# Calculate per-class accuracy
all_preds = []
all_labels = []

for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)
    all_preds.extend(predicted.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

class_acc = class_accuracy(all_preds, all_labels, len(train_dataset.classes))

for ax in axes.flatten():
    idx = np.random.randint(len(test_dataset))
    image, true_label = test_dataset[idx]
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output, 1)
    predicted_label = train_dataset.classes[predicted.item()]
    class_acc_pred = class_acc[predicted.item()]
    
    image = denormalize(image.squeeze(0), mean, std).permute(1, 2, 0).cpu().numpy()
    ax.imshow(image)
    ax.set_title(f'True: {true_label}, Pred: {predicted_label}\nAcc: {class_acc_pred:.2f}')
    ax.axis('off')

plt.tight_layout()
plt.show()
