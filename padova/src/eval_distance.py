import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.spatial import distance_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from pathlib import Path

from dataloading import get_dataloader_test
from models_binary import CNN_binary
from models import CNN
from models_ternary import CNN_ternary



def infer_matrices(model_path: Path, batch_size=256):
    dataset_folder = os.path.join('..', 'dataset', 'dataset1')
    test_loader, num_classes = get_dataloader_test(dataset_folder, batch_size)

    print("num of classes: ", int(num_classes))
    print("num samples:", len(test_loader) * batch_size)

    model_name = model_path.suffixes[-2][1:]
    print("loading model:", model_name)

    if model_name == 'full':
        model = CNN(num_classes)
        model.load_state_dict(torch.load(model_path, weights_only=False))
    elif model_name == 'bin':
        model = CNN_binary(num_classes)
        model.load_state_dict(torch.load(model_path, weights_only=False))
    elif model_name == 'ter':
        save = torch.load(model_path, weights_only=False)
        model = CNN_ternary(num_classes, delta=save['delta'])
        model.load_state_dict(save['state'])
        print('loaded delta:', save['delta'])
    else:
        raise ValueError(f"invalid model name ({model_name}) in path: {model_path}")
    
    model.eval()
    extractor = model.net

    # infer
    embeddings, labels = [], []

    with torch.no_grad():
        for X, y in tqdm(test_loader, desc="inference"):
            e = extractor(X).reshape(X.size(0), -1)

            embeddings.append(e.numpy())
            labels.append(y.numpy())

    embeddings = np.vstack(embeddings)
    labels = np.hstack(labels)

    # sort by class
    indices = np.argsort(labels)
    embeddings = embeddings[indices, :]
    labels = labels[indices]

    print(f"embeddings shape: {embeddings.shape} - labels shape: {labels.shape}")

    # compute mask_matrix, class_matrix
    print("generating class, mask matrix...")
    N = len(labels)
    mask_matrix = np.ones((N, N), dtype=np.int16)
    class_matrix = np.ones((N, N), dtype=np.int16)

    for i in range(N):
        for j in range(N):
            if labels[i] == labels[j]:
                mask_matrix[i, j] = 0
                class_matrix[i, j] = labels[i]
            else:
                class_matrix[i, j] = -1

    # compute distance matrix (4mins for 4000 embeddings)
    print("generating dist matrix...")
    dist_matrix = distance_matrix(embeddings, embeddings, p=2)

    return dist_matrix, mask_matrix, class_matrix


def generate_graphs(save_path, dist_matrix, mask_matrix, class_matrix):
    # heatmaps
    print("printing heatmaps...")
    plt.figure()
    sns.heatmap(mask_matrix)
    plt.savefig(save_path / "mask-heatmap.png", dpi=200)
    plt.close()

    plt.figure()
    sns.heatmap(class_matrix)
    plt.savefig(save_path / "class-heatmap.png", dpi=200)
    plt.close()

    plt.figure()
    sns.heatmap(dist_matrix)
    plt.savefig(save_path / "dist-heatmap.png", dpi=200)
    plt.close()

    # histogram
    print("printing histogram...")
    matching = dist_matrix[mask_matrix == 0].flatten()
    not_matching = dist_matrix[mask_matrix == 1].flatten()
    df_1 = pd.DataFrame({"distance": matching, "match_state": "match"})
    df_2 = pd.DataFrame({"distance": not_matching, "match_state": "no_match"})
    df = pd.concat((df_1, df_2))

    sns.histplot(df, x="distance", hue="match_state", stat="density", common_norm=False)
    plt.ylim((0, 0.4))
    plt.xlim((0, 55))
    plt.xlabel("Distance")
    plt.savefig(save_path / "hist-plot.png", dpi=200)
    plt.close()
    
    # ROC
    print("printing ROC...")
    y_true  = mask_matrix.flatten()
    y_score = dist_matrix.flatten()

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)

    subsample = round(len(fpr) / 500) # render only about 500 curve points
    sns.lineplot(x=fpr[::subsample], y=tpr[::subsample])
    plt.title(f"ROC Curve (auc={auc_score:.3f})")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate");
    plt.savefig(save_path / "roc-plot.png", dpi=200)
    plt.close()

    print(f"ROC AUC score: {auc_score:.3f}")
    return auc_score


def evaluate_distance(model_path: Path, load=False):
    print(f">> dist evaluating: {model_path}, load: {load}")

    dist_matrix_path = model_path.parent / 'dist_matrix_euc.pt'
    mask_matrix_path = model_path.parent / 'mask_matrix_euc.pt'
    class_matrix_path = model_path.parent / 'class_matrix_euc.pt'

    if not load:
        dist_matrix, mask_matrix, class_matrix = infer_matrices(model_path)

        torch.save(dist_matrix, dist_matrix_path)
        torch.save(mask_matrix, mask_matrix_path)
        torch.save(class_matrix, class_matrix_path)
        print("saved matrices!")
    else:
        dist_matrix = torch.load(dist_matrix_path, weights_only=False)
        mask_matrix = torch.load(mask_matrix_path, weights_only=False)
        class_matrix = torch.load(class_matrix_path, weights_only=False)
        print("loaded matrices!")
        
    print(f"matrices shape: mask {mask_matrix.shape} - class {class_matrix.shape} - dist {dist_matrix.shape}")

    auc_score = generate_graphs(model_path.parent, dist_matrix, mask_matrix, class_matrix)
    return auc_score


if __name__ == "__main__":
    ps = argparse.ArgumentParser()
    ps.add_argument('--path', type=str, default=None, help='path to the model')
    ps.add_argument('--load', type=bool, default=False, action=argparse.BooleanOptionalAction)
    args = ps.parse_args()
    
    evaluate_distance(Path(args.path), args.load)
