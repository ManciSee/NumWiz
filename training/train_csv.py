import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import ast

class HandModel(nn.Module):
    def __init__(self):
        super(HandModel, self).__init__()
        self.fc1 = nn.Linear(42, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 6)  

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Loading data
hand_data = pd.read_csv('../synthetic.csv')

# Preprocessing data
X = hand_data['landmarks'].apply(ast.literal_eval)
X = pd.DataFrame(X.tolist())
y = hand_data['class']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Converting to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values.astype(np.float32))
y_train_tensor = torch.tensor(y_train.values.astype(np.int64))
X_test_tensor = torch.tensor(X_test.values.astype(np.float32))
y_test_tensor = torch.tensor(y_test.values.astype(np.int64))

# Creating datasets and data loaders
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 32
number_workers = 0

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=number_workers)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=number_workers)

# Defining model, loss function, and optimizer
MLP_model = HandModel()
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(MLP_model.parameters(), lr=learning_rate)

# Training the model
epochs = 100
validation_loss_min = np.Inf

for epoch in range(epochs):
    training_loss = 0.0
    validation_loss = 0.0

    MLP_model.train()

    for data, label in train_loader:
        optimizer.zero_grad()
        output = MLP_model(data)

        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        training_loss += loss.item() * data.size(0)

    MLP_model.eval()

    for data, label in test_loader:
        output = MLP_model(data)
        loss = criterion(output, label)
        validation_loss += loss.item() * data.size(0)

    training_loss = training_loss / len(train_loader.dataset)
    validation_loss = validation_loss / len(test_loader.dataset)

    print(f'Epoch: {epoch+1}, Training Loss: {training_loss:.6f}, Validation Loss: {validation_loss:.6f}')
    if validation_loss <= validation_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(validation_loss_min, validation_loss))
        torch.save(MLP_model.state_dict(), 'weights/model_CSV.pth')
        validation_loss_min = validation_loss

# Testing the model
MLP_model.eval()
testing_loss = 0.0
class_correct = list(0. for i in range(6))
class_total = list(0. for i in range(6))

for data, target in test_loader:
    output = MLP_model(data)
    loss = criterion(output, target)
    testing_loss += loss.item() * data.size(0)

    _, pred = torch.max(output, 1)
    corrected = pred.eq(target.data.view_as(pred))

    for i in range(len(target)):
        label = target[i]
        class_correct[label] += corrected[i].item()
        class_total[label] += 1

testing_loss = testing_loss / len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(testing_loss))

for i in range(6):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (str(i)))

print('Test Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total), np.sum(class_correct), np.sum(class_total)))
