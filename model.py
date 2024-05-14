import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score

transform = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.ToTensor(),
])

hand_dataset = datasets.ImageFolder(root='hand_cropped', transform=transform)

# Splitting data into train and test set
train_size = int(0.8 * len(hand_dataset))
test_size = len(hand_dataset) - train_size
train_dataset, test_dataset = random_split(hand_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Meter
class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.num = 0
    
    def add(self, value, num):
        self.sum += value*num
        self.num += num
    
    def value(self):
        try:
            return self.sum/self.num
        except:
            return None

learning_rate = 0.001
epochs = 30
momentum = 0.9

# Training
def train(model, train_loader, test_loader, lr=learning_rate, epochs=epochs, momentum=momentum, logdir="logs"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    global_step = 0
    for e in range(epochs):
        print(f"Epoch {e+1}/{epochs}")
        for mode in ["train", "test"]:
            loss_meter.reset()
            acc_meter.reset()

            model.train() if mode == "train" else model.eval()
            loader = train_loader if mode == "train" else test_loader

            with torch.set_grad_enabled(mode == "train"):
                for i, (inputs, labels) in enumerate(loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if mode == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    
                    _, predicted = torch.max(outputs, 1)
                    acc = accuracy_score(labels.cpu(), predicted.cpu())
                    loss_meter.add(loss.item(), inputs.size(0))
                    acc_meter.add(acc, inputs.size(0))

                    print(f"{mode.capitalize()} Loss: {loss_meter.value()}, Accuracy: {acc_meter.value()}")
    
            torch.save(model.state_dict(),"weights/%s-%d.pth" % (model.__class__.__name__, e+1))
    return model

class HandModel(nn.Module):
    def __init__(self, in_features, hidden_units, out_classes):
        super(HandModel, self).__init__()
        self.flatten = nn.Flatten()
        self.hidden_layer = nn.Linear(in_features, hidden_units)
        self.activation = nn.Sigmoid() 
        self.output_layer = nn.Linear(hidden_units, out_classes)

    def forward(self, x):
        x = self.flatten(x)
        hidden_representation = self.hidden_layer(x)
        hidden_representation = self.activation(hidden_representation)
        scores = self.output_layer(hidden_representation)
        return scores

number_of_classes = 4
hand_classifier = HandModel(400 * 400 * 3, 128, number_of_classes)  
hand_classifier = train(hand_classifier, train_loader, test_loader, lr=learning_rate, epochs=epochs, momentum=momentum, logdir="logs")

print("Model trained successfully!")
