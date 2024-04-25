import torch
from dataloading import loadX, loadY
from models import CNN
import os
import torch.nn as nn
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_ckpt', type=str, default=None, help='Path to the checkpoint to load')
    return parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Fix the seed
torch.manual_seed(1310)

dataset_folder = os.path.join('..', 'dataset', 'dataset1')
# Read the training set
XtrainRaw1 = loadX(os.path.join(dataset_folder, 'train', r'Inertial Signals'), "train")
Ytrain1 = loadY(os.path.join(dataset_folder, "train"), "train")

# Read the test set
XtestRaw1 = loadX(os.path.join(dataset_folder, 'test', r'Inertial Signals'), "test")
Ytest1 = loadY(os.path.join(dataset_folder, "test"), "test")

# Hyperparameters
batch_size = 1000
learning_rate = 0.005
numClasses = max(Ytrain1)
print("Number of classes: ", int(numClasses))

trainData = list()
for i in range(len(XtrainRaw1)):
    sample = [XtrainRaw1[i]]
    trainData.append((torch.tensor(sample, dtype=torch.float32), Ytrain1[i] - 1))

# Create the tensor for testing
testData = list()
for i in range(len(XtestRaw1)):
    sample = [XtestRaw1[i]]
    testData.append((torch.tensor(sample, dtype=torch.float32), Ytest1[i] - 1))

# Train loader
trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=batch_size, shuffle=True)

# Instantiate the CNN
modelDataRaw_1 = CNN(numClasses).to(device)
# Setting the loss function
cost = nn.CrossEntropyLoss()
# Setting the optimizer with the model parameters and learning rate
optimizer = torch.optim.Adam(modelDataRaw_1.parameters(), lr=learning_rate)
# Define the total step to print how many steps are remaining when training
total_step = len(trainLoader)
# Number of epochs
num_epochs = 200

args = parse_args()
ckpt_path = args.load_ckpt

if ckpt_path is None:
    for epoch in range(num_epochs):
        # The train loader take one batch after the other
        for i, (signals, labels) in enumerate(trainLoader):
            signals = signals.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = modelDataRaw_1(signals)
            loss = cost(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()  # We set the previous gradient to zero: pythorch use the gradient t-1 to compute the gradient at step t
            loss.backward()
            optimizer.step()

            if (i + 1) % 11 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')

    # save the model
    torch.save(modelDataRaw_1.state_dict(), os.path.join('..', 'models', 'modelDataRaw_1.ckpt'))
    ckpt_path = os.path.join('..', 'models', 'modelDataRaw_1.ckpt')

# Load the model
modelDataRaw_1.load_state_dict(torch.load(ckpt_path))
with torch.no_grad():  # Disable the computation of the gradient
    correct = 0
    total = 0
    for signals, labels in trainLoader:  # The trainLoader is already defined for the training phase
        signals = signals.to(device)
        labels = labels.to(device)
        outputs = modelDataRaw_1(signals)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the train signals: {} %'.format(100 * correct / total))

testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=batch_size, shuffle=True)

with torch.no_grad():  # Disable the computation of the gradient
    correct = 0
    total = 0
    # Iterate for each signal
    for signals, labels in testLoader:
        signals = signals.to(device)
        labels = labels.to(device)
        outputs = modelDataRaw_1(signals)  # If batch_size>1, than the outputs is a matrix
        _, predicted = torch.max(outputs.data, 1)  # The maximum value corresponds to the predicted subject
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test signals: {} %'.format(100 * correct / total))
