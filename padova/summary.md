# RESULTS


## classification training

### FULL
#### best hp
bs=256, lr=0.005
#### roc auc
- euc 94.4
- cos 96.5
#### task:classification
94.17 (svm)
#### task:clustering
silhouette score: 0.176
rand score: 0.989
adj rand score: 0.580
  
### BIN
#### best hp
bs=128, lr=0.02
#### roc auc
- euc 97.7
- cos 97.7
#### task:classification
93.29 (knn)
#### task:clustering
silhouette score: 0.144
rand score: 0.996
adj rand score: 0.791
  
### TER
#### best hp
bs=128, lr=0.01, dreg=log, dmin=0, dmax=0.3, dmaxep=250
#### roc auc
- euc 96.7
- cos 96.5
#### task:classification
92.57 (svm)
#### task:clustering
silhouette score: 0.176
rand score: 0.996
adj rand score: 0.782



## triplet training, EUC dist

### FULL
#### best hp
bs=256, lr=0.001, margin=10
#### roc auc
+ euc 99.51
- cos 98.5
#### task:classification
94.52 (knn)
#### task:clustering
silhouette score: 0.274
rand score: 0.998
adj rand score: 0.890

### BIN
#### best hp
bs=128, lr=0.01, margin=1.5
#### roc auc
+ euc 99.14
- cos 98.8~
#### task:classification
93.50 (svm)
#### task:clustering
silhouette score: 0.146
rand score: 0.997
adj rand score: 0.819

### TER
#### best hp
bs=256, lr=0.01, margin=1.5, dreg=log, dmax=0.2, dmaxep=20
#### roc auc
+ euc 99.33
- cos 99.2
#### task:classification
93.58 (svm)
#### task:clustering
silhouette score: 0.167
rand score: 0.997
adj rand score: 0.825



## triplet training, COS dist

### FULL
#### best hp
bs=256, lr=0.001, margin=0.2
#### roc auc
+ cos 99.24
- euc 93.94
#### task:classification
94.10 (knn)
#### task:clustering
silhouette score: 0.268
rand score: 0.996
adj rand score: 0.788
  
### BIN
#### best hp
bs=128, lr=0.01, margin=0.01
#### roc auc
+ cos  99.12
- euc  99.09
#### task:classification
93.34 (svm)
#### task:clustering
silhouette score: 0.140
rand score: 0.996
adj rand score: 0.814

### TER
####  best hp
bs=256, lr=0.01, margin=0.008, dreg=const, dmax=50, dmaxep=20
####  roc auc
+ cos 99.37
- euc  99.33
#### task:classification
93.45 (knn)
#### task:clustering
silhouette score: 0.172
rand score: 0.996
adj rand score: 0.812
