import os

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
