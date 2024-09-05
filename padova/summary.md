# TRIPLET (ROC AUC scores)

## train on classification task

### FULL

BEST HP
bs=256, lr=0.005

ROC AUC
- euc 94.4
- cos 96.5
  
### BIN
BEST HP
bs=128, lr=0.02

ROC AUC
- euc 97.7
- cos 97.7
  
### TER
BEST HP
bs=128, lr=0.01, dreg=log, dmin=0, dmax=0.3, dmaxep=250

ROC AUC
- euc 96.7
- cos 96.5


## train on triplet task (EUC)

### FULL
BEST HP
bs=256, lr=0.001, margin=10

ROC AUC
+ euc 99.51
- cos 98.5
  
### BIN
BEST HP
bs=128, lr=0.01, margin=1.5

ROC AUC
+ euc 99.14
- cos 98.8~

### TER
BEST HP
bs=256, lr=0.01, margin=1.5 => 99.33

ROC AUC
+ euc 99.33
- cos 99.2



## train on triplet task (COS)

### FULL
BEST HP
bs=256, lr=0.001, margin=0.2

ROC AUC
+ cos 99.24
- euc 93.94
  
### BIN (redownload)
BEST HP
bs=128, lr=0.01, margin=0.04

ROC AUC
+ cos  99.12
- euc  99.09

### TER
BEST HP

ROC AUC
+ cos 
- euc  


# CLASSIFICATION TASK

## train on classification task

### FULL
BEST HP
bs=256, lr=0.005

ACCURACY
94.17 (svm)
  
### BIN
BEST HP
bs=128, lr=0.02

ACCURACY
93.29 (knn)
  
### TER
BEST HP
bs=128, lr=0.01, dreg=log, dmin=0, dmax=0.3, dmaxep=250

ACCURACY
92.57 (svm)


## train on triplet task (EUC)

### FULL
BEST HP
bs=256, lr=0.001, margin=10

ACCURACY
94.52 (knn)

### BIN
BEST HP
bs=128, lr=0.01, margin=1.5

ACCURACY
93.50 (svm)

### TER
BEST HP
bs=256, lr=0.01, margin=1.5 => 99.33

ACCURACY
93.58 (svm)




## train on triplet task (COS)

### FULL
BEST HP
bs=256, lr=0.001, margin=0.2

ACCURACY
94.10 (knn)

### BIN (redownload)
BEST HP
bs=128, lr=0.01, margin=0.01

ACCURACY

### TER
BEST HP

ACCURACY