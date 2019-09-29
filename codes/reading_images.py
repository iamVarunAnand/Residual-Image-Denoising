import numpy as np

print("Importing patches...")
X_test = np.load("Data - DnCNN-S/Validation Data/(47680x40x40x1) Original Validation Data/X_test.npy")

print("Normalizing data...")
X_test = X_test / 255
X_test = X_test.astype(np.float32)

print("Generating noisy data...")
Y_test = X_test + np.random.normal(0, 25 / 255, X_test.shape)

print("Saving data...")
np.save("Data - DnCNN-S/Validation Data/X_test.npy", X_test)
np.save("Data - DnCNN-S/Validation Data/Y_test.npy", Y_test)