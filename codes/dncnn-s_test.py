import numpy as np
import cv2
import time
import glob
import matplotlib.pyplot as plt
from keras.models import load_model
import keras.backend as K
from skimage.measure import compare_psnr

def to_tensor(img):
    if img.ndim == 2:
        return img[np.newaxis,...,np.newaxis]
    elif img.ndim == 3:
        return np.moveaxis(img,2,0)[...,np.newaxis]

def from_tensor(img):
    return np.squeeze(np.moveaxis(img[...,0],0,-1))

def show(x,title=None,cbar=False,figsize=None):
    plt.figure(figsize=figsize)
    plt.imshow(x,interpolation='nearest',cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()

def sum_squared_error(y_true, y_pred):
    #return K.mean(K.square(y_pred - y_true), axis=-1)
    #return K.sum(K.square(y_pred - y_true), axis=-1)/2
    return K.sum(K.square(y_pred - y_true))/2

model = load_model('Models - DnCNN-S\model_050.hdf5', custom_objects={'sum_squared_error': sum_squared_error})

file_list = glob.glob("Data - DnCNN-S/Test Data/(12x256x256x1) Original Test Data - Set 12" + "/*.png")
count = 0
for file in file_list:
    X = np.array(cv2.imread(file, cv2.IMREAD_GRAYSCALE))
    X = X.astype(np.float32) / 255.0
    Y = X + np.random.normal(0, 25, X.shape) / 255.0
    tensorY = to_tensor(Y)

    start_time = time.time()
    output = model.predict(tensorY)
    end_time = time.time()

    tensor_output = from_tensor(output)

    cv2.imwrite("Denoised"+str(count+1)+".png", np.hstack((Y*255, tensor_output*255)))
    count = count + 1

    print('Total time taken to denoise = %0.2f' % (end_time - start_time))
    PSNR = compare_psnr(X, tensor_output)
    print('PSNR = %0.2f' % PSNR)




