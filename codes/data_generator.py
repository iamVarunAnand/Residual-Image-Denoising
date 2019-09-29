import numpy as np
import matplotlib.pyplot as plt

print('Importing Data...')
src_dir = "D:\Coding\PycharmProjects\Image Processing Project\Data - DnCNN-B\Train Data\Patched Data.npy"

X_train = []
X_test = []

print('Splitting Data...')
for i in range(data.shape[0]):
    if i < 0.8 * data.shape[0]:
       X_train.append(data[i])
    else:
       X_test.append(data[i])

X_train = np.array(X_train)
X_test  = np.array(X_test)

X_test  = X_test.astype(np.float32)
X_test  = X_test / 255

print('Saving Data...')
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
