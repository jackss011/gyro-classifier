import torch
import os
import torch.nn as nn
import torchaudio
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import delta_regimes
import utils

from dataloading import loadX, loadY
from models_ternary import CNN_ternary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_ckpt', type=str, default=None, help='Path to the checkpoint to load')

    parser.add_argument('--wd', type=float, default=0.0001)
    parser.add_argument('--dreg', type=str, choices=delta_regimes.all_names)
    parser.add_argument('--dmin', type=float, default=0)
    parser.add_argument('--dmax', type=float)
    parser.add_argument('--dmaxep', type=int, default=100)
    return parser.parse_args()

args = parse_args()


num_epochs = 500
layer_inflation = 1 # all model x2, no effect, all model /2 -1.5%
# delta = 0.1 #0.01 #same as #0.1 #0.2 ~same with more zeros (80%) 
# SNR = 15
# Hyperparameters
momentum = 0.9
batch_size = 200
learning_rate = 0.01

# weight_decay = 0.0001 # 0.001
# delta_regime_type = "linear"
# delta_regime_mim = 0
# delta_regime_max = 0.3
# delta_regime_max_epoch = 50

weight_decay = args.wd
delta_regime_type = args.dreg
delta_regime_min = args.dmin
delta_regime_max = args.dmax
delta_regime_max_epoch = args.dmaxep

# create delta regime class
DeltaRegimeClass = delta_regimes.by_name(delta_regime_type)
delta_regime = DeltaRegimeClass(delta_regime_min, delta_regime_max, max_at_epoch=delta_regime_max_epoch)

hparams = dict(bs=batch_size, lr=learning_rate, m=momentum, wd=weight_decay, dreg=delta_regime.name, dmin=delta_regime.min, dmax=delta_regime.max, dmaxep=delta_regime.max_at_epoch)

# training paths
exp_name = 'ternary'
time_name = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
hparam_name = utils.hparams_to_folder(hparams)
folder = f"{exp_name}/{time_name}/{hparam_name}"

logs_path = os.path.join('./logs/', folder)
save_path = os.path.join('./results/', folder)
                         
os.makedirs(logs_path)
os.makedirs(save_path)

# Random seed
# torch.manual_seed(1310)

# choose device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setup tensorboard
writer = SummaryWriter(log_dir=logs_path)

# dataset stuff
dataset_folder = os.path.join('..', 'dataset', 'dataset1')
# Read the training set
XtrainRaw1 = loadX(os.path.join(dataset_folder, 'train', r'Inertial Signals'), "train")
Ytrain1 = loadY(os.path.join(dataset_folder, "train"), "train")

# Read the test set
XtestRaw1 = loadX(os.path.join(dataset_folder, 'test', r'Inertial Signals'), "test")
Ytest1 = loadY(os.path.join(dataset_folder, "test"), "test")

# determine num classes
num_classes = int(max(Ytrain1))
print("Number of classes: ", num_classes)


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
model = CNN_ternary(num_classes, delta=delta_regime.get(0), layer_inflation=layer_inflation).to(device)

# Setting the loss function
cost = nn.CrossEntropyLoss()

# Setting the optimizer with the model parameters and learning rate
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

# Define the total step to print how many steps are remaining when training
total_step = len(trainLoader)

# Number of epochs
best_test = 0.0

if args.load_ckpt is not None:
    model.load_state_dict(torch.load(args.load_ckpt))
else:
    ckpt_path = save_path + '/model_best.ckpt'

# main loop
for epoch in range(0, num_epochs):
    # training
    epoch_loss = 0.0

    delta = delta_regime.get(epoch)
    model.set_delta(delta)
    writer.add_scalar("DELTA", delta, epoch)

    for i, (signals, labels) in enumerate(trainLoader):
        signals: torch.Tensor = signals.to(device)
        labels: torch.Tensor = labels.to(device)

        # normalize
        # signals = torch.nn.functional.normalize(signals, p=1.0, dim=(-1, -2))

        # add noise
        # noise = torch.rand_like(signals, device=device) - 0.5
        # snr = torch.ones((signals.size(0), 1, 6), device=device) * SNR
        # # print(signals.size(), noise.size())
        # signals = torchaudio.functional.add_noise(signals, noise, snr=snr)

        # Forward pass
        outputs = model(signals)
        loss = cost(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()

        # logging
        loss_item = loss.item()
        epoch_loss += loss_item

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

    writer.add_scalar("TRAIN LOSS", epoch_loss/len(trainLoader), epoch)

    # if epoch == 150:
    #     for group in optimizer.param_groups:
    #         group['lr'] /= 10
    #         group['weight_decay'] /= 10

    #     print(*optimizer.param_groups)

    # if epoch == 225:
    #     for group in optimizer.param_groups:
    #         group['lr'] /= 10
    #         # group['weight_decay'] /= 10


    # once every n epochs
    if epoch % 10 == 0:
        # track entropy of model weights
        weights_entropy = model.get_weights_entropy()
        writer.add_scalar("WEIGHTS ENTROPY", weights_entropy, epoch)

        # validation on training
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

        # validation on test
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

        # print zeros and +-1 weights
        z, p1, m1, num = model.stats()
        z_perc, p1_perc, m1_perc = (z/num*100, p1/num*100, m1/num*100)
        writer.add_scalar("weights/zeros", z_perc, epoch)
        writer.add_scalar("weights/plus-one", p1_perc, epoch)
        writer.add_scalar("weights/minus-one", m1_perc, epoch)


writer.add_hparams(
    hparams,
    {'hparam/best-test-acc': best_test}
)

writer.close()
