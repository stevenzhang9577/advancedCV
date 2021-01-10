import numpy as np
import keras
from keras.datasets import cifar100

def get_mean_std(images):
    mean_channels = []
    std_channels = []

    for i in range(images.shape[-1]):
        mean_channels.append(np.mean(images[:, :, :, i]))
        std_channels.append(np.std(images[:, :, :, i]))

    return mean_channels, std_channels

def pre_processing(train_images, test_images):
    images = np.concatenate((train_images, test_images), axis = 0)
    mean, std = get_mean_std(images)

    for i in range(test_images.shape[-1]):
        train_images[:, :, :, i] = (train_images[:, :, :, i] - mean[i]) / std[i]
        test_images[:, :, :, i] = (test_images[:, :, :, i] - mean[i]) / std[i]
    
    return train_images, test_images


def get_cifar_gen():
    # get dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(
        label_mode='fine')

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.

   # y_train = tf.keras.utils.to_categorical(y_train)
    #y_test = tf.keras.utils.to_categorical(y_test)
    x_test = np.load('cifar100_validation_x_full.npy')
    y_test = np.load('cifar100_validation_y_full.npy')

    # preprocess data
    x_train, x_test = pre_processing(x_train, x_test)
   # datagen = ImageDataGenerator(
    #    shear_range=0.2,
     #   zoom_range=0.2,
      #  horizontal_flip=True
    #)
    #cifar_gen = datagen.flow(x_train, y_train, batch_size=batch_size)

    #testgen = ImageDataGenerator()
    #cifar_test_gen = testgen.flow(x_test, y_test, batch_size=batch_size)

    return x_test, y_test

#(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# Normalize data.
#X_train = X_train.astype('float32') / 255
#X_test = X_test.astype('float32') / 255
#X_train_mean = np.mean(X_train, axis=0)
#X_test-=X_train_mean
#print(Y_test)
#print(Y_test.shape)
#Y_test = keras.utils.to_categorical(Y_test, 10)

X_test2 = np.load('test/bim_cifar100_image_validation_vgg19_pro.npy')
#X_test2-=X_train_mean
print(X_test2.shape)
#print(X_test2)
X_test,Y_test=get_cifar_gen()
Y_test2=np.load('test/bim_cifar100_label_validation_vgg19_pro.npy')
#Y_test2=Y_test2.reshape(Y_test2.shape[0], 1)
print(Y_test2)
#Y_test2 = keras.utils.to_categorical(Y_test2, 10)



import random
L1 = random.sample(range(0, 5000), 2500)
L2=random.sample(range(0,5000),2500)

for i in range(2500):
    X_test[L1[i]]=X_test2[L2[i]]
    Y_test[L1[i]]=Y_test2[L2[i]]

print(X_test)

np.savez('data/cifar100_validation_combined_bim_vgg19.npz',X_test,Y_test)
origin_model_path = '../vgg19_dense.h5'
origin_model = keras.models.load_model(origin_model_path)
Y_test = keras.utils.to_categorical(Y_test,100)
loss2,acc2=origin_model.evaluate(X_test,Y_test)
print('loss:',loss2)
print('acc:',acc2)