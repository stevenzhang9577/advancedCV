import numpy as np
import pandas as pd
import keras


def get_cifar10_vgg16_test():
    from keras.datasets import cifar10
    subtract_pixel_mean = False
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

    # Normalize data.
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    # If subtract pixel mean is enabled
    if subtract_pixel_mean:
        X_train_mean = np.mean(X_train, axis=0)
        X_train -= X_train_mean
        X_test -= X_train_mean
    # Y_test = keras.utils.to_categorical(Y_test, 10)
    return X_test,Y_test

def get_cifar10_resnet20_test():
    from keras.datasets import cifar10
    subtract_pixel_mean = False
    (_, _), (X_test, Y_test) = cifar10.load_data()
    #X_train = np.load('./data/cifar10_train.npz')['x']
    # Y_train = np.load('./data/cifar10_validation.npz')['y']
    # Normalize data.
    #X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    # If subtract pixel mean is enabled
    # if subtract_pixel_mean:
    #     X_train_mean = np.mean(X_train, axis=0)
    #     X_train -= X_train_mean
    #     X_test -= X_train_mean
    # Y_test = keras.utils.to_categorical(Y_test, 10)
    return X_test,Y_test


def get_cifar10_vgg16():
    X_test = np.load('cifar10_test_x_full.npy')
    Y_test = np.load('cifar10_test_y_full.npy')
    return X_test,Y_test


def get_cifar10_vgg16_test_csv():
    return pd.read_csv('.\\extracted_feature\\cifar10_vgg16\\cifar10_test.csv')

def get_cifar10_vgg16_validation_csv():
    return pd.read_csv('.\\extracted_feature\\cifar10_vgg16\\cifar10_validation.csv')




def get_cifar10_vgg16_test_csv_75():
    return pd.read_csv('.\\extracted_feature\\cifar10_vgg16_0.7515\\vgg16_cifar10_feature_test.csv')

def get_cifar10_vgg16_validation_csv_75():
    return pd.read_csv('.\\extracted_feature\\cifar10_vgg16_0.7515\\vgg16_cifar10_feature_validation.csv')

def get_cifar10_vgg16_test_csv_70():
    return pd.read_csv('.\\extracted_feature\\cifar10_vgg16_0.7000\\vgg16_cifar10_feature_test.csv')

def get_cifar10_vgg16_validation_csv_70():
    return pd.read_csv('.\\extracted_feature\\cifar10_vgg16_0.7000\\vgg16_cifar10_feature_validation.csv')



def get_cifar10_resnet20_test_csv_80():
    return pd.read_csv('.\\extracted_feature\\cifar10_resnet20_0.8000\\cifar10_resnet20_feature_test.csv')

def get_cifar10_resnet20_validation_csv_80():
    return pd.read_csv('.\\extracted_feature\\cifar10_resnet20_0.8000\\cifar10_resnet20_feature_validation.csv')


def get_cifar10_vgg16_test_csv_85():
    return pd.read_csv('.\\extracted_feature\\cifar10_vgg16_0.85\\cifar10_vgg16_feature_test.csv')

def get_cifar10_vgg16_validation_csv_85():
    return pd.read_csv('.\\extracted_feature\\cifar10_vgg16_0.85\\cifar10_vgg16_feature_validation.csv')



def get_test_csv(exp_id):
    exp_model_dict = {
                      'cifar10_vgg16_0.75':get_cifar10_vgg16_test_csv_75,
                      'cifar10_vgg16_0.70': get_cifar10_vgg16_test_csv_70,
                      'cifar10_resnet20_0.80':get_cifar10_resnet20_test_csv_80,
                      'cifar10_vgg16':get_cifar10_vgg16_test_csv,
                      'cifar10_vgg16_85': get_cifar10_vgg16_test_csv_85
                      }
    return exp_model_dict[exp_id]

def get_validate_csv(exp_id):
    exp_model_dict = {
                      'cifar10_vgg16_0.75': get_cifar10_vgg16_validation_csv_75,
                      'cifar10_vgg16_0.70': get_cifar10_vgg16_validation_csv_70,
                      'cifar10_resnet20_0.80': get_cifar10_resnet20_validation_csv_80,
                      'cifar10_vgg16': get_cifar10_vgg16_validation_csv,
                      'cifar10_vgg16_85':get_cifar10_vgg16_validation_csv_85
                      }
    return exp_model_dict[exp_id]

def get_data(exp_id):
    exp_model_dict = {
                      'cifar10_vgg16_0.75':get_cifar10_vgg16_test,
                      'cifar10_vgg16_0.70':get_cifar10_vgg16_test,
                      'cifar10_resnet20_0.80':get_cifar10_vgg16_test,
                      'cifar10_vgg16': get_cifar10_vgg16,
                      'cifar10_vgg16_85':get_cifar10_vgg16
                     }
    return exp_model_dict[exp_id]

def get_model(exp_id):
    if exp_id == 'cifar10_vgg16_0.75':
        origin_model_path = '../model/cifar10_vgg16_0.7640.h5'
        origin_model = keras.models.load_model(origin_model_path)
    elif exp_id == 'cifar10_vgg16_0.70':
        origin_model_path = '../model/cifar10_vgg16_0.7070.h5'
        origin_model = keras.models.load_model(origin_model_path)
    elif exp_id == 'cifar10_resnet20_0.80':
        origin_model_path = '../model/cifar10_ResNet20v1_model.091.h5'
        origin_model = keras.models.load_model(origin_model_path)
    elif exp_id == 'cifar10_vgg16':
        origin_model_path = '../model/cifar10_vgg16.h5'
        origin_model = keras.models.load_model(origin_model_path)
    elif exp_id == 'cifar10_vgg16_85':
        origin_model_path = '../model/cifar10_vgg16_0.8503.h5'
        origin_model = keras.models.load_model(origin_model_path)
    return origin_model


def get_state_number(exp_id):
    exp_model_dict = {
                      'cifar10_vgg16':17,
                      'cifar10_vgg16_0.75':81,
                      'cifar10_vgg16_0.70':81,
                      'cifar10_resnet20_0.80':317,
                      'cifar10_vgg16_85':396
                      }
    return exp_model_dict[exp_id]


def return_test_images(test_images):
    return test_images


#X_train_cifar10_resnet20 = np.load('./data/cifar10_train.npz')['x']
#Y_train_cifar10_resnet20 = np.load('./data/cifar10_train.npz')['y']
#X_train_cifar10_resnet20 = X_train_cifar10_resnet20.astype('float32') / 255
#X_train_cifar10_resnet20_mean = np.mean(X_train_cifar10_resnet20, axis=0)
def preprocess_cifar10_resnet20(test_images):
    subtract_pixel_mean = True
    # Normalize data.
    # X_train = X_train_cifar10_resnet20.astype('float32') / 255
    # If subtract pixel mean is enabled
    if subtract_pixel_mean:
        # x_train_mean = X_train_cifar10_resnet20_mean
        test_images -= X_train_cifar10_resnet20_mean
    return test_images


def data_proprecessing(exp_id):
    if exp_id == 'cifar10_resnet20_0.80':
        return preprocess_cifar10_resnet20
    else:
        return return_test_images

