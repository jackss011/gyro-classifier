import os
import torch
import numpy as np

def loadX(path: str, data_split: str):
    """Load the signals of accelerometer and gyroscope.
    Given the path of X,Y,Z components, create a vector with the components.

    Args:
        path (str): Path where the dataset is located
        dataset (str): can be train or test
    Returns:
        Matrix with all the components: [[[accX_1][accY_1][accZ_1][gyrX_1][gyrY_1][gyrZ_1]], ..., [[accX_n][accY_n][accZ_n][gyrX_n][gyrY_n][gyrZ_n]]]
    """
    #Open the accelerometer's files
    fileAccX = open(os.path.join(path, f"{data_split}_acc_x.txt"),"r")
    fileAccY = open(os.path.join(path, f"{data_split}_acc_y.txt"),"r")
    fileAccZ = open(os.path.join(path, f"{data_split}_acc_z.txt"),"r")
    #Open the gyrpscope's files
    fileGyrX = open(os.path.join(path, f"{data_split}_gyr_x.txt"),"r")
    fileGyrY = open(os.path.join(path, f"{data_split}_gyr_y.txt"),"r")
    fileGyrZ = open(os.path.join(path, f"{data_split}_gyr_z.txt"),"r")

    signals=list()
    #For each signal, decomposed in the three components, build the final matrix
    for compAccX,compAccY,compAccZ,compGyrX,compGyrY,compGyrZ in zip(fileAccX,fileAccY,fileAccZ,fileGyrX,fileGyrY,fileGyrZ):
        #Convert to float
        compAccX=[float(x) for x in compAccX.split()]
        compAccY=[float(y) for y in compAccY.split()]
        compAccZ=[float(z) for z in compAccZ.split()]
        compGyrX=[float(x) for x in compGyrX.split()]
        compGyrY=[float(y) for y in compGyrY.split()]
        compGyrZ=[float(z) for z in compGyrZ.split()]
        #Add them to the matrix of signals
        signals.append([compAccX, compAccY, compAccZ,compGyrX, compGyrY,compGyrZ])
    return signals

def loadY(path: str, data_split: str):
    """Load the labels corresponding the signals.

    Args:
        path (str): Path where the dataset is located
        dataset (str): can be train or test
    Returns:
        vectors of labels
    """
    fileLables = open(os.path.join(path, f"y_{data_split}.txt"),"r")
    return [int(v) for v in fileLables]


def get_dataloader_train(dataset_folder, batch_size):
    # Read the training set
    XtrainRaw1 = loadX(os.path.join(dataset_folder, 'train', r'Inertial Signals'), "train")
    Ytrain1 = loadY(os.path.join(dataset_folder, "train"), "train")

    numClasses = max(Ytrain1)

    trainData = list()
    for i in range(len(XtrainRaw1)):
        sample = [XtrainRaw1[i]]
        trainData.append((torch.tensor(sample, dtype=torch.float32), Ytrain1[i] - 1))

    # Train loader
    trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=batch_size, shuffle=False)
    return trainLoader, numClasses


def get_dataloader_test(dataset_folder, batch_size):
    # Read the test set
    XtestRaw1 = loadX(os.path.join(dataset_folder, 'test', r'Inertial Signals'), "test")
    Ytest1 = loadY(os.path.join(dataset_folder, "test"), "test")
    numClasses = max(Ytest1)

    # Create the tensor for testing
    testData = list()
    for i in range(len(XtestRaw1)):
        sample = [XtestRaw1[i]]
        testData.append((torch.tensor(sample, dtype=torch.float32), Ytest1[i] - 1))

    # Test loader
    testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=batch_size, shuffle=False)
    return testLoader, numClasses


class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder: str, *, train: bool):
        self.dataset_folder = dataset_folder
        self.is_train = train

        which_ds = "train" if self.is_train else "test"

        X_raw = loadX(os.path.join(dataset_folder, which_ds, 'Inertial Signals'), which_ds)
        y_raw = loadY(os.path.join(dataset_folder, which_ds), which_ds)

        self.num_classes = max(y_raw)
        self.X = [torch.tensor([x], dtype=torch.float32) for x in X_raw]
        self.y = [y - 1 for y in y_raw]
        self.labels = np.array(self.y, dtype=np.int16)

        assert(len(self.X) == len(self.y))

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index: int):
        anchor = self.X[index]
        anchor_label = self.y[index]

        pos_filter = (self.labels == anchor_label)
        pos_filter[index] = 0 # avoid picking same item
        pos_indices = pos_filter.nonzero()[0]
        pos_index = np.random.choice(pos_indices)
        pos = self.X[pos_index]
        pos_label = self.y[pos_index]

        neg_indices = (self.labels != anchor_label).nonzero()[0]
        neg_index = np.random.choice(neg_indices)
        neg = self.X[neg_index]
        neg_label = self.y[neg_index]

        return anchor, pos, neg, anchor_label, pos_label, neg_label
