import torch
from dataloading import loadX, loadY
from models_binary import CNN_binary
import os
import torch.nn as nn
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_ckpt', type=str, default=None, help='Path to the checkpoint to load')
    return parser.parse_args()


# choose device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# save path
exp_name = 'binary1'
time_name = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
logs_path = os.path.join('./logs/', f"{exp_name}__{time_name}")
save_path = os.path.join('./results/', f"{exp_name}__{time_name}")
os.makedirs(logs_path)
os.makedirs(save_path)

# Random seed
# torch.manual_seed(1310)

# Setup tensorboard
writer = SummaryWriter(log_dir=logs_path)

dataset_folder = os.path.join('..', 'dataset', 'dataset1')
# Read the training set
XtrainRaw1 = loadX(os.path.join(dataset_folder, 'train', r'Inertial Signals'), "train")
Ytrain1 = loadY(os.path.join(dataset_folder, "train"), "train")

# Read the test set
XtestRaw1 = loadX(os.path.join(dataset_folder, 'test', r'Inertial Signals'), "test")
Ytest1 = loadY(os.path.join(dataset_folder, "test"), "test")

# Hyperparameters
batch_size = 200
learning_rate = 0.01
numClasses = max(Ytrain1)
print("Number of classes: ", int(numClasses))

# Create the tensor for training
trainData = list()
for i in range(len(XtrainRaw1)):
    sample = [XtrainRaw1[i]]
    trainData.append((torch.tensor(sample, dtype=torch.float32), Ytrain1[i] - 1))

# Create the tensor for testing
testData = list()
for i in range(len(XtestRaw1)):
    sample = [XtestRaw1[i]]
    testData.append((torch.tensor(sample, dtype=torch.float32), Ytest1[i] - 1))



# Train data loader
trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=batch_size, shuffle=True)

# Test data loader
testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=batch_size, shuffle=True)

# Instantiate the CNN
model = CNN_binary(numClasses).to(device)

# Setting the loss function
cost = nn.CrossEntropyLoss()

# Setting the optimizer with the model parameters and learning rate
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Define the total step to print how many steps are remaining when training
total_step = len(trainLoader)

# Number of epochs
num_epochs = 1000
best_test = 0.0

args = parse_args()
if args.load_ckpt is not None:
    model.load_state_dict(torch.load(args.load_ckpt))
else:
    ckpt_path = save_path + '/model_best.ckpt'

# main loop
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (signals, labels) in enumerate(trainLoader):
        signals = signals.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(signals)
        loss = cost(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()

        # logging
        loss_item = loss.item()
        running_loss += loss_item

        # STE/clamping
        for p in list(model.parameters()):
            if hasattr(p, 'org'):
                p.data.copy_(p.org)
        optimizer.step()
        for p in list(model.parameters()):
            if hasattr(p, 'org'):
                p.org.copy_(p.data.clamp_(-1, 1))

        if (i + 1) % 21 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss_item:.4f}')

    writer.add_scalar("TRAIN LOSS", running_loss/len(trainLoader), epoch)

    with torch.no_grad():  # Disable the computation of the gradient
        correct = 0
        total = 0
        for signals, labels in trainLoader:  # The trainLoader is already defined for the training phase
            signals = signals.to(device)
            labels = labels.to(device)
            outputs = model(signals)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        print('TRAIN ACC: {} %'.format(train_acc))
        writer.add_scalar("TRAIN ACC", train_acc, epoch)

    with torch.no_grad():  # Disable the computation of the gradient
        correct = 0
        total = 0
        # Iterate for each signal
        for signals, labels in testLoader:
            signals = signals.to(device)
            labels = labels.to(device)
            outputs = model(signals)  # If batch_size>1, than the outputs is a matrix
            _, predicted = torch.max(outputs.data, 1)  # The maximum value corresponds to the predicted subject
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_acc = 100 * correct / total
        print('TEST ACC: {} %'.format(test_acc))
        writer.add_scalar("TEST ACC", test_acc, epoch)

        if best_test < (100 * correct / total):
            best_test = 100 * correct / total
            torch.save(model.state_dict(), ckpt_path)
            print('Best checkpoint saved!')
        print('BEST TEST ACC: {} %'.format(best_test))
        writer.add_scalar("BEST TEST ACC", best_test, epoch)

writer.close()

# model.net(signals).reshape(batch_size,-1).sign()

