import torch
from dataloading import loadX, loadY
from models_binary import CNN_binary, CNN_binary_relu, init_weights
import os
import torch.nn as nn
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_ckpt', type=str, default=None, help='Path to the checkpoint to load')
    parser.add_argument('--bs', type=int, help="batch size")
    parser.add_argument('--lr', type=float, help="learning rate")
    parser.add_argument('--af32', type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--drop', type=float, default=0)
    parser.add_argument('--full-fc', type=bool, default=False, action=argparse.BooleanOptionalAction)
    return parser.parse_args()

args = parse_args()

num_epochs = 500
# Hyperparameters
# batch_size = 200
# learning_rate = 0.01
exp_name = 'binary'

batch_size = args.bs
learning_rate =  args.lr
# lr_steps = [150, 300]
lr_steps = []
lr_gamma = 0.1
af32 = args.af32
dropout = args.drop
model_name = args.model
full_fc = args.full_fc

hparams = dict(bs=batch_size, lr=learning_rate)
if af32:
    hparams['af32'] = 'Y'
if model_name:
    hparams['model'] = model_name
if dropout > 0:
    hparams['drop'] = dropout
if full_fc:
    hparams['full-fc'] = 'Y'

# choose binary model
ModelSelected = None
if not model_name:
    ModelSelected = CNN_binary
elif model_name == 'relu':
    ModelSelected = CNN_binary_relu
else:
    raise ValueError("No such --model as:", model_name)

# choose device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# save path
# exp_name = 'binary-sched'
time_name = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
hparam_name = utils.hparams_to_folder(hparams)
folder = f"{exp_name}/{time_name}/{hparam_name}"

logs_path = os.path.join('./logs/', folder)
save_path = os.path.join('./results/', folder)
os.makedirs(logs_path)
os.makedirs(save_path)

# Random seed
torch.manual_seed(1310)

# Setup tensorboard
writer = SummaryWriter(log_dir=logs_path)

dataset_folder = os.path.join('..', 'dataset', 'dataset1')
# Read the training set
XtrainRaw1 = loadX(os.path.join(dataset_folder, 'train', r'Inertial Signals'), "train")
Ytrain1 = loadY(os.path.join(dataset_folder, "train"), "train")

# Read the test set
XtestRaw1 = loadX(os.path.join(dataset_folder, 'test', r'Inertial Signals'), "test")
Ytest1 = loadY(os.path.join(dataset_folder, "test"), "test")

numClasses = max(Ytrain1)

print("model: ", ModelSelected)
print("number of classes: ", int(numClasses))
print("LR steps: ", lr_steps)
print("LR gamma: ", lr_gamma)
print("hparams:", hparams)

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


trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=batch_size, shuffle=True)
testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=batch_size, shuffle=True)

# Instantiate the CNN
model_kwargs=dict(af32=af32, dropout=dropout, full_fc=full_fc)
model = ModelSelected(numClasses, **model_kwargs).to(device)
print("fc type", type(model.fc))

if af32:
    init_weights(model)

# Setting the loss function
cost = nn.CrossEntropyLoss()

# Setting the optimizer with the model parameters and learning rate
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=120, gamma=0.1)

# Define the total step to print how many steps are remaining when training
total_step = len(trainLoader)

best_test = 0.0

if args.load_ckpt is not None:
    model.load_state_dict(torch.load(args.load_ckpt))
else:
    ckpt_path = save_path + '/best.bin.ckpt'


# main loop
for epoch in range(num_epochs):
    # +++++ TRAIN ++++++
    model.train()
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

        # PRINT
        if (i + 1) % 21 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss_item:.4f}')

    # TENSORBOARD
    writer.add_scalar("TRAIN LOSS", running_loss/len(trainLoader), epoch)
    writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)

    # SCALE LR
    if epoch in lr_steps:
        print('scaling LR by factor of 10')
        for group in optimizer.param_groups:
            group['lr'] *= lr_gamma
            group['weight_decay'] *= lr_gamma # not used

    # +++++ EVAL ++++++ (every 10)
    if epoch % 10 == 0:
        # => train accuracy
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

        # => test accuracy
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
                save = {'state': model.state_dict(), **model_kwargs}
                torch.save(save, ckpt_path)
                print('Best checkpoint saved!')
            print('BEST TEST ACC: {} %'.format(best_test))
            writer.add_scalar("BEST TEST ACC", best_test, epoch)

writer.close()
