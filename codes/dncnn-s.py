import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from keras.layers import Conv2D, Activation, BatchNormalization, Input, Subtract
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import keras.backend as K

X_train = np.load('X_train.npy')

def DnCNN_S(depth):

    inpt = Input(shape = (None, None, 1))
    x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same', kernel_initializer = 'orthogonal')(inpt)
    x = Activation(activation = 'relu')(x)

    for i in range(depth - 2):
        x = Conv2D(filters = 64, kernel_size = (3, 3), strides=(1, 1), padding='same', kernel_initializer='orthogonal')(x)
        x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)

    x = Conv2D(filters = 1, kernel_size = (3, 3), strides = (1, 1), padding = 'same', kernel_initializer = 'orthogonal')(x)
    x = Subtract()([inpt, x])

    model = Model(inputs = inpt, outputs = x)

    print(model.summary)
    return model

def data_generator(sigma, epochs, batch_size):

    for i in range(0, epochs):
        for j in range(0, X_train.shape[0] // batch_size, batch_size):
            X_train_batch = X_train[j:j+batch_size]
            noise = np.random.normal(0, sigma, X_train_batch.shape)
            Y_train_batch = X_train_batch + noise

            yield Y_train_batch, X_train_batch

def lr_schedule(epoch):

    initial_lr = 0.001

    if epoch <= 30:
        lr = initial_lr
    elif epoch <= 60:
        lr = initial_lr/10
    elif epoch <= 80:
        lr = initial_lr/20
    else:
        lr = initial_lr/20

    print('current learning rate is %2.8f' %lr)

    return lr

def find_initial_epoch():
    file_list = glob.glob('model_*.hdf5')
    initial_epoch = 0

    if file_list:
        epochs_finished = []
        for file_ in file_list:
            result = re.findall("model_(.*).hdf5.*",file_)
            epochs_finished.append(int(result[0]))
        initial_epoch = max(epochs_finished)

    return initial_epoch

def sum_squared_error(y_true, y_pred):
    #return K.mean(K.square(y_pred - y_true), axis=-1)
    #return K.sum(K.square(y_pred - y_true), axis=-1)/2
    return K.sum(K.square(y_pred - y_true))/2

print("Setting up Model...")
model = DnCNN_S(17)

checkpoint = ModelCheckpoint("model_{epoch:03d}.hdf5", save_weights_only = False, period = 1, verbose = 1)
lr_scheduler = LearningRateScheduler(lr_schedule)

initial_epoch = find_initial_epoch()

if(initial_epoch > 0):
    print('resuming by loading epoch %03d' % initial_epoch)
    model = load_model('model_%03d.hdf5' % initial_epoch, compile = False)

model.compile(loss = sum_squared_error, optimizer = Adam(0.001), metrics = ['mse'])

print("Training Model...")
History = model.fit_generator(data_generator(sigma = 25, epochs = 300, batch_size = 64), steps_per_epoch = 2980, epochs = 300,
                              callbacks = [checkpoint, lr_scheduler], initial_epoch = initial_epoch)

plt.plot(History.history['mean_squared_error'])
plt.title('LOSS VS EPOCH')
plt.ylabel('Mean_Squared_Error')
plt.xlabel('Epoch')
plt.show()


