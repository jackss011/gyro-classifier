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

if __name__ == '__main__':
    # ========> DEVICE <=========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("using device:", torch.cuda.get_device_name())
    else:
        print("using device:", "cpu")

    # ========> HPARAM <=========
    epochs = 10
    batch_size = 256
    lr = 0.001
    margin = 1.5

    hparams = dict(lr=lr, bs=batch_size, margin=margin)

    # ========> LOGGING <=========
    time_folder = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    hparam_folder = utils.hparams_to_folder(hparams)
    exp_folder = f"triplet/{time_folder}/{hparam_folder}"

    log_folder = Path('./logs/')    / exp_folder
    res_folder = Path('./results/') / exp_folder
    log_folder.mkdir(parents=True, exist_ok=True)
    res_folder.mkdir(parents=True, exist_ok=True)

    summary = SummaryWriter(log_folder)

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


    # ========> TRAINING <=========
    for e in tqdm(range(1, epochs + 1), desc="epochs"):
        model.train()

        running_loss = []
        for anchor, pos, neg, anchor_label, _, _ in tqdm(train_loader, desc="steps", leave=False):
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
        print(">> loss:", mean_loss)
