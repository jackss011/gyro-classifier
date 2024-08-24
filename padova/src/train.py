import torch
from dataloading import loadX, loadY
from models import CNN
import os
import torch.nn as nn
import argparse
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datetime import datetime
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_ckpt', type=str, default=None, help='Path to the checkpoint to load')
    parser.add_argument('--bs', type=int, help="batch size")
    parser.add_argument('--lr', type=float, help="learning rate")
    return parser.parse_args()

args = parse_args()


num_epochs = 500
# Hyperparameters
batch_size = args.bs
learning_rate =  args.lr
# batch_size = 1000
# learning_rate = 0.005

hparams = dict(bs=batch_size, lr=learning_rate)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setup tensorboard
exp_name = 'full'
time_name = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
hparam_name = utils.hparams_to_folder(hparams)
folder = f"{exp_name}/{time_name}/{hparam_name}"

logs_path = os.path.join('./logs/', folder)
save_path = os.path.join('./results/', folder)

writer = SummaryWriter(log_dir=logs_path)

# Fix the seed
torch.manual_seed(1310)

# LOAD DATA
dataset_folder = os.path.join('..', 'dataset', 'dataset1')
# Read the training set
XtrainRaw1 = loadX(os.path.join(dataset_folder, 'train', r'Inertial Signals'), "train")
Ytrain1 = loadY(os.path.join(dataset_folder, "train"), "train")

# Read the test set
XtestRaw1 = loadX(os.path.join(dataset_folder, 'test', r'Inertial Signals'), "test")
Ytest1 = loadY(os.path.join(dataset_folder, "test"), "test")

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

# Test loader
testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=batch_size, shuffle=True)

# MODEL
# Instantiate the CNN
model = CNN(numClasses).to(device)
# Setting the loss function
cost = nn.CrossEntropyLoss()
# Setting the optimizer with the model parameters and learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Define the total step to print how many steps are remaining when training
total_step = len(trainLoader)
# Number of epochs

if args.load_ckpt is not None:
    model.load_state_dict(torch.load(args.load_ckpt))
else:
    ckpt_path = save_path + '/model_best.ckpt'


best_test = 0.0

print("# params: ", utils.count_parameters(model))

for epoch in range(num_epochs):
    epoch_loss = 0
    model.train()
    # The train loader take one batch after the other
    for i, (signals, labels) in enumerate(trainLoader):
        signals = signals.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(signals)
        loss = cost(outputs, labels)

        loss_item = loss.item()
        epoch_loss += loss_item

        # Backward and optimize
        optimizer.zero_grad()  # We set the previous gradient to zero: pythorch use the gradient t-1 to compute the gradient at step t
        loss.backward()
        optimizer.step()

        if (i + 1) % 11 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss_item:.5f}')

    writer.add_scalar("TRAIN LOSS", epoch_loss/len(trainLoader), epoch)
    
    # once every n epochs
    if epoch % 10 == 0:
        # validation on training
        with torch.no_grad():  # Disable the computation of the gradient
            model.eval()

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

        # validation on test
        with torch.no_grad():  # Disable the computation of the gradient
            model.eval()

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
                print('Better model found!')
            print('BEST TEST ACC: {} %'.format(best_test))
            writer.add_scalar("BEST TEST ACC", best_test, epoch)
            
writer.close()
    # save the model
    # torch.save(modelDataRaw_1.state_dict(), os.path.join('..', 'models', 'modelDataRaw_1.ckpt'))
    # ckpt_path = os.path.join('..', 'models', 'modelDataRaw_1.ckpt')

# Load the model
# model.load_state_dict(torch.load(ckpt_path))
# with torch.no_grad():  # Disable the computation of the gradient
#     correct = 0
#     total = 0
#     for signals, labels in trainLoader:  # The trainLoader is already defined for the training phase
#         signals = signals.to(device)
#         labels = labels.to(device)
#         outputs = model(signals)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     train_acc = 100 * correct / total
#     print('Accuracy of the network on the train signals: {}%'.format(train_acc))
#     writer.add_scalar("TRAIN ACC", train_acc, epoch)


# with torch.no_grad():  # Disable the computation of the gradient
#     correct = 0
#     total = 0
#     # Iterate for each signal
#     for signals, labels in testLoader:
#         signals = signals.to(device)
#         labels = labels.to(device)
#         outputs = model(signals)  # If batch_size>1, than the outputs is a matrix
#         _, predicted = torch.max(outputs.data, 1)  # The maximum value corresponds to the predicted subject
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     test_acc = 100 * correct / total

#     print('Accuracy of the network on the test signals: {} %'.format(test_acc))
#     writer.add_scalar("TEST ACC", test_acc, epoch)
