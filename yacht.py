import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

# Load data
data = pd.read_csv("yacht_hydrodynamics.data", sep=" ", names=["Long Position", "Prismatic coefficient", "length-displacement ratio", "bean-draught ratio", "length-bean ratio", "froude number", "residuary resistance"], error_bad_lines=False)
data = data.fillna(0)

# Convert data to tensors
features = torch.tensor(data[["Long Position", "Prismatic coefficient", "length-displacement ratio", "bean-draught ratio", "length-bean ratio", "froude number"]].values)
target = minmax_scale(data[["residuary resistance"]], feature_range=(0, 1), axis=0, copy=True)
target = torch.tensor(target)

# Split data into training and testing sets
train_features, test_features, train_target, test_target = train_test_split(features, target, test_size=0.2, random_state=100)

# Define a custom dataset class
class DatasetT(Dataset):
    def __init__(self, features, labels):
        self.X = features
        self.Y = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        return x, y

# Create data loaders
trainset = DatasetT(train_features, train_target)
testset = DatasetT(test_features, test_target)
train_loader = DataLoader(dataset=trainset, batch_size=277, shuffle=True)
test_loader = DataLoader(dataset=testset, batch_size=62, shuffle=True)

# Define the neural network model
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 12)
        self.fc2 = nn.Linear(12, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model
model = Network()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optm = optim.Adam(model.parameters(), lr=0.0003)

# Train the model
train_mse = []
test_mse = []
for epoch in range(500):
    yPred = []
    yA = []
    for i, (x, y) in enumerate(train_loader):
        optm.zero_grad()                   
        outputs = model(x.float())                        
        loss = criterion(outputs, y.float())            
        loss.backward()                         
        optm.step() 
        yPred.append(outputs)
        yA.append(y)
    train_mse.append(mean_squared_error(torch.cat(yA).detach().numpy(), torch.cat(yPred).detach().numpy()))
    yPred = []
    yA = []
    with torch.no_grad():
        for x,y in test_loader:
            output = model(x.float()) 
            yPred.append(output)
            yA.append(y)
    test_mse.append(mean_squared_error(torch.cat(yA).detach().numpy(), torch.cat(yPred).detach().numpy()))
    
# Function to predict yacht hydrodynamics
def predict_yacht_hydrodynamics(long_position, prismatic_coefficient, length_displacement_ratio, bean_draught_ratio, length_bean_ratio, froude_number):
    input_values = torch.tensor([[long_position, prismatic_coefficient, length_displacement_ratio, bean_draught_ratio, length_bean_ratio, froude_number]])
    output = model(input_values.float())
    return output.detach().numpy()[0][0]

# Get user input
long_position = float(input("Enter the long position: "))
prismatic_coefficient = float(input("Enter the prismatic coefficient: "))
length_displacement_ratio = float(input("Enter the length-displacement ratio: "))
bean_draught_ratio = float(input("Enter the bean-draught ratio: "))
length_bean_ratio = float(input("Enter the length-bean ratio: "))
froude_number = float(input("Enter the Froude number: "))

# Predict yacht hydrodynamics
predicted_resistance = predict_yacht_hydrodynamics(long_position, prismatic_coefficient, length_displacement_ratio, bean_draught_ratio, length_bean_ratio, froude_number)

print("Predicted residuary resistance: ", predicted_resistance)