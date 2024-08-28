import torch
from dataloading import TripletDataset
from models_binary import CNN_binary, CNN_binary_relu, init_weights
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import utils


# ========> DEVICE <=========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("using device:", torch.cuda.get_device_name())
else:
    print("using device:", "cpu")

# ========> ARGS <=========
ps = argparse.ArgumentParser()
ps.add_argument('--bs', type=int, help="batch size")
ps.add_argument('--lr', type=float, help="learning rate")
ps.add_argument('--margin', type=float, default=1.5, help="triplet loss margin")
args = ps.parse_args()

# ========> HPARAMS <=========
epochs = 500
batch_size = args.bs    # 256
lr = args.lr            # 0.001
margin = args.margin    # 1.5

hparams = dict(lr=lr, bs=batch_size, margin=margin)
print("hyper-parameters:", hparams)

# ========> LOGGING <=========
time_folder = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
hparam_folder = utils.hparams_to_folder(hparams)
exp_folder = f"triplet/{time_folder}/{hparam_folder}"

log_folder = Path('./logs/')    / exp_folder
res_folder = Path('./results/') / exp_folder
log_folder.mkdir(parents=True, exist_ok=True)
res_folder.mkdir(parents=True, exist_ok=True)

summary = SummaryWriter(log_folder)
ckpt_file = res_folder / "best_model.pth"

# ========> DATASET <=========
dataset_folder = Path("..") / "dataset" / "dataset1"
train_ds = TripletDataset(dataset_folder, train=True)
test_ds  = TripletDataset(dataset_folder, train=False)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, )
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=True, )

num_classes = train_ds.num_classes
print("num classes:", train_ds.num_classes)

# ========> MODEL <=========
model = CNN_binary(num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

distance_fn = torch.nn.PairwiseDistance(p=2)
criterion = torch.nn.TripletMarginWithDistanceLoss(distance_function=distance_fn, margin=margin, swap=True, reduction="mean")

val_distance_fn = torch.nn.PairwiseDistance(p=2)
val_criterion = torch.nn.TripletMarginWithDistanceLoss(distance_function=val_distance_fn, margin=margin, swap=False, reduction="mean")

loweset_validation_loss = float('inf')

for e in tqdm(range(1, epochs + 1), desc="epochs"):
    # ========> TRAINING <=========
    model.train()

    running_loss = []
    for anchor, pos, neg, _, _, _ in tqdm(train_loader, desc="batches", leave=False):
        anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)

        optimizer.zero_grad()
        anchor_out = model.net(anchor)
        pos_out = model.net(pos)
        neg_out = model.net(neg)

        loss = criterion(anchor_out, pos_out, neg_out)
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())

    mean_loss = np.array(running_loss).mean()
    summary.add_scalar("train/loss", mean_loss, e)
    print(">> training loss:", mean_loss)

    # ========> VALIDATION <=========
    if e % 10 == 0: # every 10 epoch
        # ==> on training set <==
        with torch.no_grad():
            model.eval()

            running_loss = []
            for anchor, pos, neg, _, _, _ in tqdm(train_loader, desc="validation (traning set)", leave=False):
                anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)

                anchor_out = model.net(anchor)
                pos_out = model.net(pos)
                neg_out = model.net(neg)

                loss = val_criterion(anchor_out, pos_out, neg_out)
                running_loss.append(loss.item())

            mean_loss = np.array(running_loss).mean()
            summary.add_scalar("validation/loss_training", mean_loss, e)
            print(">> validation loss (training set):", mean_loss)

    if e % 1 == 0: # every 10 epochs
        # ==> on test set <==
        with torch.no_grad():
            model.eval()

            running_loss = []
            for anchor, pos, neg, _, _, _ in tqdm(test_loader, desc="validation (test set)", leave=False):
                anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)

                anchor_out = model.net(anchor)
                pos_out = model.net(pos)
                neg_out = model.net(neg)

                loss = val_criterion(anchor_out, pos_out, neg_out)
                running_loss.append(loss.item())

            mean_loss = np.array(running_loss).mean()
            summary.add_scalar("validation/loss_test", mean_loss, e)
            print(">> validation loss (test set):", mean_loss)

            if mean_loss < loweset_validation_loss:
                loweset_validation_loss = mean_loss
                print(">> best model found")
                torch.save(model.state_dict(), ckpt_file)

    print("\n") # some space between epochs

