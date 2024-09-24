import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import argparse
import json

import utils
from dataloading import TripletDataset, OpenTripletDataset
from models import CNN
import models_binary
from models_binary import CNN_binary
from binary_modules import BinaryHammingLoss
import delta_regimes
import models_ternary
from models_ternary import CNN_ternary, TernaryHammingLoss
from eval_distance import evaluate_distance
from eval_tasks import eval_clustering



# ========> DEVICE <=========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("using device:", torch.cuda.get_device_name())
else:
    print("using device:", "cpu")

# ========> ARGS <=========
ps = argparse.ArgumentParser()
ps.add_argument('--model', type=str, choices=['full', 'bin', 'ter'], help="which model", required=True)
ps.add_argument('--epochs', type=int, default=100, help="number of training epochs")
ps.add_argument('--bs', type=int, help="batch size", required=True)
ps.add_argument('--lr', type=float, help="learning rate", required=True)
ps.add_argument('--margin', type=float, default=1.5, help="triplet loss margin")
ps.add_argument('--dist', type=str, choices=['euc', 'cos', 'hamm'], default='euc', help='distance function')
ps.add_argument('--eval', type=bool, default=True, action=argparse.BooleanOptionalAction, help="calculate roc score and print graphs")
# ternary delta regime specific
ps.add_argument('--dreg', type=str,   default="const", choices=delta_regimes.all_names, help="delta regime curve")
ps.add_argument('--dmin', type=float, default=0, help="delta at epoch 0")
ps.add_argument('--dmaxep', type=int, default=50, help="epoch at which delta reaches dmax")
ps.add_argument('--dmax', type=float, default=0.2, help="delta at epoch dmaxep")
ps.add_argument('--open', type=int, default=None, help="use open set dataset on selected split")
args = ps.parse_args()

# ========> HPARAMS <=========
model_name = args.model
epochs = args.epochs    # 100
batch_size = args.bs    # 256
lr = args.lr            # 0.01
margin = args.margin    # 1.5
distance_name = args.dist
open_set = args.open is not None
open_set_split = args.open

if distance_name == 'hamm':
    assert(model_name in ['bin', 'ter'])

hparams = dict(model=model_name, epochs=epochs, lr=lr, bs=batch_size, margin=margin, dist=distance_name)

# add delta regime hparams
if model_name == 'ter':
    hparams |= dict(dreg=args.dreg, dmin=args.dmin, dmaxep=args.dmaxep, dmax=args.dmax)
    DeltaRegimeClass = delta_regimes.by_name(args.dreg)
    delta_regime = DeltaRegimeClass(args.dmin, args.dmax, max_at_epoch=args.dmaxep)
else:
    delta_regime = None

if open_set:
    assert(open_set_split >= 0 and open_set_split < 10)
    hparams |= dict(open=open_set_split)

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
if not open_set:
    train_ds = TripletDataset(dataset_folder, train=True)
    test_ds  = TripletDataset(dataset_folder, train=False)
else:
    train_ds = OpenTripletDataset(dataset_folder, train=True, split_num=open_set_split)
    test_ds = OpenTripletDataset(dataset_folder, train=False, split_num=open_set_split)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, )
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=True, )

num_classes = train_ds.num_classes
print("num classes:", train_ds.num_classes)

if open_set:
    print("selected labels:", train_ds.test_labels)
    print("train labels:", set(train_ds.labels))
    print("test labels:", set(test_ds.labels))
    print(f"train samples: {len(train_ds)} | test samples: {len(test_ds)}")
    assert(len(set(train_ds.labels).intersection(set(test_ds.labels))) == 0)

# ========> MODEL <=========
if model_name == 'full':
    model = CNN(num_classes).to(device)
elif model_name == 'bin':
    model = CNN_binary(num_classes).to(device)
    models_binary.init_weights(model)
elif model_name == 'ter':
    model = CNN_ternary(num_classes, delta=delta_regime.get(0)).to(device)
    models_ternary.init_weights(model)
else:
    raise ValueError(f"invalid model name: {model_name}")

extract_features = lambda x: model.net(x).reshape(x.size(0), -1)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=15)

# choose distance functions
if distance_name == 'euc':
    distance_fn = torch.nn.PairwiseDistance(p=2)
    val_distance_fn = torch.nn.PairwiseDistance(p=2)
elif distance_name == 'cos':
    distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
    val_distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
elif distance_name == 'hamm':
    if model_name == 'bin':
        distance_fn = BinaryHammingLoss()
        val_distance_fn = BinaryHammingLoss()
    elif model_name == 'ter':
        distance_fn = TernaryHammingLoss(delta=delta_regime.get(0))
        val_distance_fn = TernaryHammingLoss(delta=delta_regime.get(0))
    else:
        raise ValueError(f"cannot use `hamm` distance with model: {model_name}")
else:
    raise ValueError(f"invalid distance function name {distance_name}")

criterion = torch.nn.TripletMarginWithDistanceLoss(distance_function=distance_fn, margin=margin, swap=True, reduction="mean")
val_criterion = torch.nn.TripletMarginWithDistanceLoss(distance_function=val_distance_fn, margin=margin, swap=False, reduction="mean")

lowest_validation_loss = float('inf')

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
        running_loss.append(loss.item())

        # binary/ternary: load 'org' weights
        if model_name in ['bin', 'ter']:
            for p in list(model.parameters()):
                if hasattr(p, 'org'):
                    p.data.copy_(p.org)

        optimizer.step()

        # binary/ternary: save clamped weights to 'org'
        if model_name in ['bin', 'ter']:
            for p in list(model.parameters()):
                if hasattr(p, 'org'):
                    p.org.copy_(p.data.clamp_(-1, 1))

    # log
    mean_loss = np.array(running_loss).mean()
    summary.add_scalar("train/loss", mean_loss, e)
    print(">> training loss:", mean_loss)

    # ========> VALIDATION <=========
    with torch.no_grad():
        model.eval()

        running_loss = []
        running_pos_dist = []
        running_neg_dist = []
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

            running_pos_dist.append(pos_dist.mean().item())
            running_neg_dist.append(neg_dist.mean().item())

            count += pos_dist.size(0)
            correct += torch.sum(pos_dist < neg_dist).item()

        # loss
        mean_loss = np.array(running_loss).mean()
        summary.add_scalar("validation/loss_test", mean_loss, e)
        print(">> validation loss (test set):", mean_loss)

        # distances
        mean_pos_dist = np.array(running_pos_dist).mean()
        mean_neg_dist = np.array(running_neg_dist).mean()
        summary.add_scalar("inspection/mean_distance_ap", mean_pos_dist, e)
        summary.add_scalar("inspection/mean_distance_an", mean_neg_dist, e)
        summary.add_scalar("inspection/mean_distance_difference", mean_neg_dist - mean_pos_dist, e)

        # accuracy
        perc_accuracy = correct / count
        summary.add_scalar("validation/accuracy", perc_accuracy, e)
        print(f">> accuracy (test set): {perc_accuracy*100:.1f}")

        # checkpoint
        if mean_loss < lowest_validation_loss:
            lowest_validation_loss = mean_loss

            save = {'state': model.state_dict() }
            if model_name == 'ter':
                save |= {'delta': model._delta}
            if open_set:
                save |= {'open': open_set_split}

            torch.save(save, ckpt_file)
            print(f">> best model found, (delta: {save.get('delta')}), (open split: {save.get('open')})")
        
        # lr scheduler
        scheduler.step(mean_loss)
        summary.add_scalar("lr", scheduler.get_last_lr()[0], e)

        # delta scheduler
        if model_name == 'ter':
            new_delta = delta_regime.get(e)
            model.set_delta(new_delta)

            if distance_name == 'hamm':
                distance_fn.delta = new_delta
                val_distance_fn.delta = new_delta

            summary.add_scalar("ternary/delta", model._delta, e)
            zeros, _, _, total = model.weight_count()
            summary.add_scalar("ternary/sparsity", zeros/total*100, e)

    print("\n") # some space between epochs


# ========> EVALUATION <=========
if args.eval:
    print("\n\n==== Evaluation ====")

    # evaluate using euc distance
    euc_auc_score = evaluate_distance(ckpt_file, no_save=True, distance_fn="euc", gen_graphs=False)
    summary.add_scalar("eval/roc_auc_score_euc", euc_auc_score * 100, epochs)

    # evaluate using cos distance
    cos_auc_score = evaluate_distance(ckpt_file, no_save=True, distance_fn="cos", gen_graphs=False)
    summary.add_scalar("eval/roc_auc_score_cos", cos_auc_score * 100, epochs)

    if model_name in ['bin', 'ter']:
        # evaluate using hamm distance
        hamm_auc_score = evaluate_distance(ckpt_file, no_save=True, distance_fn="hamm", gen_graphs=False)
        summary.add_scalar("eval/roc_auc_score_hamm", hamm_auc_score * 100, epochs)
    else:
        hamm_auc_score = None

    if open_set:
        sh_score, rand_score, adj_rand_score = eval_clustering(ckpt_file)

        row = {
            'dist': distance_name,
            'model': model_name,
            'hp': utils.hparams_to_folder(hparams),
            'split': open_set_split,
            'roc_euc': float(euc_auc_score),
            'roc_cos': float(cos_auc_score),
            'roc_hamm': float(hamm_auc_score) if hamm_auc_score is not None else None,
            'silh': float(sh_score),
            'rand': float(rand_score),
            'rand_adj': float(adj_rand_score),
        }

        res_file = Path('./logs-triplet/') / 'open-set-results.jsonl'
        with open(res_file, 'a', encoding='UTF-8') as f:
            f.write(json.dumps(row) + '\n')

summary.close()

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
