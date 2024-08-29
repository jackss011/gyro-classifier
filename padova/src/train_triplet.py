import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import argparse

import models_binary
from models_binary import CNN_binary
from models import CNN
from dataloading import TripletDataset
import utils
from eval_distance import evaluate_distance



# ========> DEVICE <=========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("using device:", torch.cuda.get_device_name())
else:
    print("using device:", "cpu")

# ========> ARGS <=========
ps = argparse.ArgumentParser()
ps.add_argument('--model', type=str, choices=['full', 'bin'], help="which model", required=True)
ps.add_argument('--epochs', type=int, default=100, help="number of training epochs")
ps.add_argument('--bs', type=int, help="batch size", required=True)
ps.add_argument('--lr', type=float, help="learning rate", required=True)
ps.add_argument('--margin', type=float, default=1.5, help="triplet loss margin")
ps.add_argument('--dist', type=str, choices=['euc', 'cos'], default='euc', help='distance function')
ps.add_argument('--eval', type=bool, default=True, action=argparse.BooleanOptionalAction)
args = ps.parse_args()

# ========> HPARAMS <=========
model_name = args.model
epochs = args.epochs    # 100
batch_size = args.bs    # 256
lr = args.lr            # 0.01
margin = args.margin    # 1.5
distance_name = args.dist

hparams = dict(model=model_name, lr=lr, bs=batch_size, margin=margin, dist=distance_name)
print("hyper-parameters:", hparams)

# ========> LOGGING <=========
time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
exp_folder = f"{model_name}/{time}/{utils.hparams_to_folder(hparams)}"

log_folder = Path('./logs-triplet/')     / exp_folder
res_folder = Path('./results-triplet/')  / exp_folder
log_folder.mkdir(parents=True, exist_ok=True)
res_folder.mkdir(parents=True, exist_ok=True)

summary = SummaryWriter(log_folder)
ckpt_file = res_folder / f"best.{model_name}.pth"

# ========> DATASET <=========
dataset_folder = Path("..") / "dataset" / "dataset1"
train_ds = TripletDataset(dataset_folder, train=True)
test_ds  = TripletDataset(dataset_folder, train=False)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, )
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=True, )

num_classes = train_ds.num_classes
print("num classes:", train_ds.num_classes)

# ========> MODEL <=========
if model_name == 'full':
    model = CNN(num_classes).to(device)
elif model_name == 'bin':
    model = CNN_binary(num_classes).to(device)
    models_binary.init_weights(model)
else:
    raise ValueError(f"invalid model name: {model_name}")

extract_features = lambda x: model.net(x).reshape(x.size(0), -1)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=15)

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
        anchor_out = extract_features(anchor)
        pos_out = extract_features(pos)
        neg_out = extract_features(neg)
        assert(anchor_out.ndim == 2)

        loss = criterion(anchor_out, pos_out, neg_out)
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())

    mean_loss = np.array(running_loss).mean()
    summary.add_scalar("train/loss", mean_loss, e)
    print(">> training loss:", mean_loss)

    # ========> VALIDATION <=========
    with torch.no_grad():
        model.eval()

        running_loss = []
        count = 0
        correct = 0
        for anchor, pos, neg, _, _, _ in tqdm(test_loader, desc="validation (test set)", leave=False):
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)

            anchor_out = extract_features(anchor)
            pos_out = extract_features(pos)
            neg_out = extract_features(neg)

            loss = val_criterion(anchor_out, pos_out, neg_out)
            running_loss.append(loss.item())

            pos_dist = val_distance_fn(anchor_out, pos_out)
            neg_dist = val_distance_fn(anchor_out, neg_out)

            count += pos_dist.size(0)
            correct += torch.sum(pos_dist < neg_dist).item()

        mean_loss = np.array(running_loss).mean()
        summary.add_scalar("validation/loss_test", mean_loss, e)
        print(">> validation loss (test set):", mean_loss)

        perc_accuracy = correct / count
        summary.add_scalar("validation/accuracy", perc_accuracy, e)
        print(f">> accuracy (test set): {perc_accuracy*100:.1f}")

        if mean_loss < loweset_validation_loss:
            loweset_validation_loss = mean_loss
            print(">> best model found")
            torch.save(model.state_dict(), ckpt_file)
        
        scheduler.step(mean_loss)
        summary.add_scalar("lr", scheduler.get_last_lr()[0], e)

    print("\n") # some space between epochs


# ========> EVALUATION <=========
if args.eval:
    print("\n\n==== Evaluation ====")
    evaluate_distance(ckpt_file)


#  if e % 10 == 0: # every 10 epoch
#         # ==> on training set <==
#         with torch.no_grad():
#             model.eval()

#             running_loss = []
#             for anchor, pos, neg, _, _, _ in tqdm(train_loader, desc="validation (traning set)", leave=False):
#                 anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)

#                 anchor_out = model.net(anchor).reshape(anchor.size(0), -1)
#                 pos_out = model.net(pos).reshape(pos.size(0), -1)
#                 neg_out = model.net(neg).reshape(neg.size(0), -1)

#                 loss = val_criterion(anchor_out, pos_out, neg_out)
#                 running_loss.append(loss.item())

#             mean_loss = np.array(running_loss).mean()
#             summary.add_scalar("validation/loss_training", mean_loss, e)
#             print(">> validation loss (training set):", mean_loss)
