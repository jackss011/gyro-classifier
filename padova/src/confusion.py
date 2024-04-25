import torch
from dataloading import loadX, loadY
from models_binary import CNN_binary, CNN_binary_notlast
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


def hamming_distance(a, b):
    r = (1 << np.arange(8))[:, None]
    return np.count_nonzero((np.bitwise_xor(a, b) & r) != 0)/len(a)


def euc_distance(v1, v2):
    dist = [(a - b)**2 for a, b in zip(v1, v2)]
    return math.sqrt(sum(dist))/len(v1)

'''
# dataset path
dataset_folder = os.path.join('..', 'dataset', 'dataset1')

# Read the training set
XtrainRaw1 = loadX(os.path.join(dataset_folder, 'train', r'Inertial Signals'), "train")
Ytrain1 = loadY(os.path.join(dataset_folder, "train"), "train")

# Read the test set
XtestRaw1 = loadX(os.path.join(dataset_folder, 'test', r'Inertial Signals'), "test")
Ytest1 = loadY(os.path.join(dataset_folder, "test"), "test")

# number of classes
numClasses = max(Ytrain1)
print("Number of classes: ", int(numClasses))

# Create the tensor for training
trainData = list()
for i in range(len(XtrainRaw1)):
    sample = [XtrainRaw1[i]]
    trainData.append((torch.tensor(sample, dtype=torch.float32), Ytrain1[i] - 1))
bs_tr = len(trainData)
trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=bs_tr, shuffle=True)

# Create the tensor for testing
testData = list()
for i in range(len(XtestRaw1)):
    sample = [XtestRaw1[i]]
    testData.append((torch.tensor(sample, dtype=torch.float32), Ytest1[i] - 1))
bs_te = len(testData)
testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=bs_te, shuffle=True)

# model paths
model_bin_path = './results/model_best_binary.ckpt'
model_lastnonbinary_path = './results/model_best_lastnonbinary.ckpt'

# load the model
model = CNN_binary_notlast(numClasses)
model.load_state_dict(torch.load(model_lastnonbinary_path))
# model = CNN_binary(numClasses)
# model.load_state_dict(torch.load(model_bin_path))
feature_extractor = model.net

for signals, labels in testLoader:
    features = feature_extractor(signals).reshape(bs_te, -1)
    # bin_features = torch.where(features >= 0, 1., -1.)

# bin_features = torch.where(bin_features.double() < 1.0, 0.0, bin_features.double()).to(torch.int32)
labels = labels.to(torch.int32)

sorted_labels, index_labels = torch.sort(labels)
sorted_features = features[index_labels]
# sorted_bin_features = bin_features[index_labels]

dist_matrix_euc = torch.zeros(bs_te, bs_te)
mask_matrix_euc = torch.ones(bs_te, bs_te)
# dist_matrix_bin = torch.zeros(bs_te, bs_te)
# mask_matrix_bin = torch.ones(bs_te, bs_te)

for i in range(bs_te):
    for j in range(bs_te):
        print(str(i) + ' / ' + str(j))
        # dist_matrix_bin[i][j] = hamming_distance(sorted_bin_features[i], sorted_bin_features[j])
        dist_matrix_euc[i][j] = euc_distance(sorted_features[i], sorted_features[j])
        if sorted_labels[i] == sorted_labels[j]:
            # mask_matrix_bin[i][j] = 0
            mask_matrix_euc[i][j] = 0

# dist_matrix_bin = dist_matrix_bin.to(torch.float16)
# mask_matrix_bin = mask_matrix_bin.to(torch.float16)
dist_matrix_euc = dist_matrix_euc.to(torch.float16)
mask_matrix_euc = mask_matrix_euc.to(torch.float16)
torch.save(dist_matrix_euc, 'dist_matrix_euc.pt')
torch.save(mask_matrix_euc, 'mask_matrix_euc.pt')
# torch.save(dist_matrix_bin, 'dist_matrix_bin.pt')
# torch.save(mask_matrix_bin, 'mask_matrix_bin.pt')

'''

# dist_matrix_bin = torch.load('dist_matrix_bin.pt')
# mask_matrix_bin = torch.load('mask_matrix_bin.pt')
dist_matrix_euc = torch.load('dist_matrix_euc.pt')
mask_matrix_euc = torch.load('mask_matrix_euc.pt')

# dist_matrix = dist_matrix_bin
# mask_matrix = mask_matrix_bin
dist_matrix = dist_matrix_euc
mask_matrix = mask_matrix_euc
matching = dist_matrix[mask_matrix == 0]
matching = matching[matching != 0].numpy()
non_matching = dist_matrix[mask_matrix == 1].numpy()
# np.random.shuffle(non_matching)
# non_matching = non_matching[0:len(matching)]

# histogram
bin_n = 150
# plt.hist(matching, bin_n)
# plt.hist(non_matching, bin_n)
# plt.savefig('hist_' + str(bin_n) + 'bins.png', dpi=200)
plt.hist(matching, bin_n, weights=np.ones_like(matching)/float(len(matching)))
plt.hist(non_matching, bin_n, weights=np.ones_like(non_matching)/float(len(non_matching)))
plt.savefig('histnorm_' + str(bin_n) + 'bins.png', dpi=200)
plt.close()
# np.concatenate([np.expand_dims(matching[0:1000], axis=1), np.expand_dims(non_matching[0:1000], axis=1)], axis=1)

# ROC curve

fpr, tpr, thresholds = roc_curve(mask_matrix.flatten().numpy(), dist_matrix.flatten().numpy())
# fnr = 1-tpr
# far = fpr
# frr = fnr
# eer = (far + frr)/2
roc_auc = roc_auc_score(mask_matrix.flatten().numpy(), dist_matrix.flatten().numpy())
plt.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig('roc.png', dpi=200)
plt.close()

print('Done!')


