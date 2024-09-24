import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.spatial import distance_matrix
from sklearn.metrics import roc_auc_score, roc_curve, pairwise_distances
from pathlib import Path

from dataloading import ListDataset, OpenListDataset
from models_binary import CNN_binary
from models import CNN
from models_ternary import CNN_ternary


def infer_embeddings(model_path: Path, batch_size=256, train_ds=False, quantize=False):
    # handle different save types
    save = torch.load(model_path, weights_only=False)
    open_set = False
    if type(save) == dict:
        state = save.pop('state')
        open_set_split = save.pop('open', None)
        open_set = open_set_split is not None
        inits = save.copy()

        print("model init parameteres (fron dict):", inits)
        if open_set:
            print("doing open set on split num:", open_set_split)
    elif type(save) == list:
        state, inits = save
        print("model init parameteres (fron list):", inits)
    else:
        state = save
        inits = {}

    # dataset/loader stuff
    dataset_folder = os.path.join('..', 'dataset', 'dataset1')
    if not open_set:
        ds = ListDataset(dataset_folder, train=train_ds)
    else:
        ds = OpenListDataset(dataset_folder, train=train_ds, split_num=open_set_split)
        print("test labels:", ds.test_labels)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    num_classes = ds.num_classes
    print("num of classes: ", int(num_classes))
    print("num samples:", len(ds))

    # load model
    model_name = model_path.suffixes[-2][1:]
    print("loading model:", model_name)
        
    if model_name == 'full':
        model = CNN(num_classes, **inits)
    elif model_name == 'bin':
        model = CNN_binary(num_classes, **inits)
    elif model_name == 'ter':
        assert('delta' in inits)
        model = CNN_ternary(num_classes, **inits)
        print('loaded delta:', model._delta)
    else:
        raise ValueError(f"invalid model name ({model_name}) in path: {model_path}")
    
    model.load_state_dict(state)
    model.eval()
    extractor = model.net

    # inference
    embeddings, labels = [], []

    with torch.no_grad():
        for X, y in tqdm(loader, desc="inference"):
            e = extractor(X).reshape(X.size(0), -1)

            embeddings.append(e.numpy())
            labels.append(y.numpy())

    embeddings = np.vstack(embeddings)
    labels = np.hstack(labels)

    # sort by class
    indices = np.argsort(labels)
    embeddings = embeddings[indices, :]
    labels = labels[indices]

    if quantize:
        if model_name == 'full':
            raise ValueError(f"cant quantize full precision model")
        elif model_name == 'bin':
            embeddings = np.where(embeddings >= 0, 1, -1).astype(np.int16)
            print("quantized (bin)!")
        elif model_name == 'ter':
            assert('delta' in inits)
            delta = inits['delta']
            embeddings = np.where(embeddings >= delta, 1, np.where(embeddings <= -delta, -1, 0)).astype(np.int16)
            print("quantized (ter)!")
        else:
            raise ValueError(f"invalid model name ({model_name}) in path: {model_path}")

    print(f"embeddings shape: {embeddings.shape} - labels shape: {labels.shape}")
    return embeddings, labels



def infer_matrices(model_path: Path, distance_fn="euc"):
    do_quantize = (distance_fn == 'hamm') # only quantize if hamming distance
    embeddings, labels = infer_embeddings(model_path, train_ds=False, quantize=do_quantize)

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
    print(f"generating dist matrix ({distance_fn}, quantize={do_quantize})...")

    if distance_fn == 'euc':
        dist_matrix = pairwise_distances(embeddings, metric='euclidean')
        # dist_matrix = distance_matrix(embeddings, embeddings, p=2)
    elif distance_fn == 'cos':
        norms = np.linalg.norm(embeddings, axis=1)
        norms = norms.reshape((-1, 1)) @ norms.reshape((1, -1))
        dist_matrix = 1.0 - (embeddings @ embeddings.T / norms)
    elif distance_fn == 'hamm':
        assert(np.sum(embeddings == 1) + np.sum(embeddings == 0) + np.sum(embeddings == -1) == embeddings.size)
        dist_matrix = pairwise_distances(embeddings, metric='hamming')
    else:
        raise ValueError(f"invalid distance function name {distance_fn}")

    return dist_matrix, mask_matrix, class_matrix



def generate_graphs(save_path, dist_matrix, mask_matrix, class_matrix, fpr, tpr, auc_score, tag="euc"):
    suffix = "_" + tag

    # heatmaps
    print("printing heatmaps...")
    plt.figure()
    sns.heatmap(mask_matrix)
    plt.savefig(save_path / f"mask-heatmap.png", dpi=200)
    plt.close()

    plt.figure()
    sns.heatmap(class_matrix)
    plt.savefig(save_path / f"class-heatmap.png", dpi=200)
    plt.close()

    plt.figure()
    sns.heatmap(dist_matrix)
    plt.savefig(save_path / f"dist-heatmap{suffix}.png", dpi=200)
    plt.close()

    # histogram
    print("printing histogram...")
    matching = dist_matrix[mask_matrix == 0].flatten()
    not_matching = dist_matrix[mask_matrix == 1].flatten()
    df_1 = pd.DataFrame({"distance": matching, "match_state": "match"})
    df_2 = pd.DataFrame({"distance": not_matching, "match_state": "no_match"})
    df = pd.concat((df_1, df_2))

    sns.histplot(df, x="distance", hue="match_state", stat="density", common_norm=False)
    # plt.ylim((0, 0.4))
    # plt.xlim((0, 55))
    plt.xlabel(f"Distance ({tag})")
    plt.savefig(save_path / f"hist-plot{suffix}.png", dpi=200)
    plt.close()
    
    # ROC
    try:
        print("printing ROC...")
        subsample = round(len(fpr) / 500) + 1 # render only about 500 curve points
        sns.lineplot(x=fpr[::subsample], y=tpr[::subsample])
        plt.title(f"ROC Curve ({tag}) [auc={auc_score*100:.1f}]")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate");
        plt.savefig(save_path / f"roc-plot{suffix}.png", dpi=200)
        plt.close()
    except:
        pass


def evaluate_distance(model_path: Path, distance_fn="euc", load=False, no_save=False, gen_graphs=True):
    print(f"\n>> dist evaluating: {model_path}, load: {load}")
    print(f"using distance:", distance_fn)

    dist_matrix, mask_matrix, class_matrix = infer_matrices(model_path, distance_fn=distance_fn)
    
    print(f"matrices shape: mask {mask_matrix.shape} - class {class_matrix.shape} - dist {dist_matrix.shape}")

    # calculate roc
    y_true  = mask_matrix.flatten()
    y_score = dist_matrix.flatten()

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)

    if gen_graphs:
        generate_graphs(model_path.parent, dist_matrix, mask_matrix, class_matrix, fpr, tpr, auc_score, tag=distance_fn)

    print(f"ROC AUC score ({distance_fn}): {auc_score*100:.1f}")
    return auc_score


if __name__ == "__main__":
    ps = argparse.ArgumentParser()
    ps.add_argument('--path', type=str, default=None, help='path to the model')
    ps.add_argument('--dist', type=str, choices=['euc', 'cos', 'hamm'], default='euc', help='distance function')
    ps.add_argument('--load', type=bool, default=False, action=argparse.BooleanOptionalAction)
    args = ps.parse_args()
    
    evaluate_distance(Path(args.path), distance_fn=args.dist, load=args.load)





# def evaluate_distance(model_path: Path, distance_fn="euc", load=False, no_save=False, gen_graphs=True):
#     print(f">> dist evaluating: {model_path}, load: {load}")
#     print(f">> using distance:", distance_fn)

#     dist_matrix_path = model_path.parent / f'dist_matrix_{distance_fn}.pt'
#     mask_matrix_path = model_path.parent / f'mask_matrix_{distance_fn}.pt'
#     class_matrix_path = model_path.parent / f'class_matrix_{distance_fn}.pt'

#     # save and load distance matrixes (not really needed)
#     if not load:
#         dist_matrix, mask_matrix, class_matrix = infer_matrices(model_path, distance_fn=distance_fn)

#         if no_save:
#             print("skipping matrix save!")
#         else:
#             torch.save(dist_matrix, dist_matrix_path)
#             torch.save(mask_matrix, mask_matrix_path)
#             torch.save(class_matrix, class_matrix_path)
#             print("saved matrices!")
#     else:
#         dist_matrix = torch.load(dist_matrix_path, weights_only=False)
#         mask_matrix = torch.load(mask_matrix_path, weights_only=False)
#         class_matrix = torch.load(class_matrix_path, weights_only=False)
#         print("loaded matrices!")
        
#     print(f"matrices shape: mask {mask_matrix.shape} - class {class_matrix.shape} - dist {dist_matrix.shape}")

#     # calculate roc
#     y_true  = mask_matrix.flatten()
#     y_score = dist_matrix.flatten()

#     fpr, tpr, thresholds = roc_curve(y_true, y_score)
#     auc_score = roc_auc_score(y_true, y_score)

#     if gen_graphs:
#         generate_graphs(model_path.parent, dist_matrix, mask_matrix, class_matrix, fpr, tpr, auc_score, tag=distance_fn)

#     return auc_score