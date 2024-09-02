# TRIPLET (ROC AUC scores)

## train on classification task
### FULL
bs=256, lr=0.005
roc auc: 94.4
### BIN
bs=128, lr=0.02
roc auc: 97.7
### TER
bs=128, lr=0.01, dreg=log, dmin=0, dmax=0.3, dmaxep=250
roc auc: 96.7

## train on triplet task
### FULL
bs=256, lr=0.001, margin=10
roc auc: 99.51
### BIN
bs=128, lr=0.01, margin=1.5
roc auc: 99.14
### TER
bs=256, lr=0.01, margin=1.5 => 99.33
roc auc: 99.33
low sparsity ~15%