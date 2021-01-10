import numpy as np
import time
import argparse
#from gluoncv.data import ImageNet
#from mxnet.gluon.data import DataLoader
#from mxnet.gluon.data.vision import transforms
from tqdm import tqdm
from keras.models import load_model, Model
from sa import fetch_dsa, fetch_lsa, get_sc
from utils import *
import os
import sys
import datetime
import keras
from keras.applications.vgg19 import VGG19
import math
from keras.models import Model
import random
from keras.datasets import mnist
from numpy import arange
import argparse
from keras.applications import vgg19,resnet50
from keras.applications.vgg19 import preprocess_input

CLIP_MIN = -0.5
CLIP_MAX = 0.5


def get_adversarial_cifar10(adversarial='fgsm'):
    X_adv = np.load('E://prioritize/input_perturbation/data/cifar10_combined_10000_image_' + str(adversarial) + '.npy')
    Y_adv = np.load('E://prioritize/input_perturbation/data/cifar10_combined_10000_label_' + str(adversarial) + '.npy')
    for i in range(len(Y_adv)):
        if type(Y_adv[i]) is np.ndarray:
            Y_adv[i] = int(Y_adv[i][0])
        else:
            Y_adv[i] = int(Y_adv[i])
    return X_adv,Y_adv

def get_imagenet():
    ft = np.load('data/imagenet-val-5000.npz')
    X_test = ft['x_test']
  #  X_test=ft['arr_0']
    Y_test = ft['y_test']
  #  Y_test=ft['arr_1']
    # print(X_test.shape)
    # X_test = X_test.reshape(X_test.shape[0], 224, 224, 1)
    # X_test /= 255
    # print(Y_test)
    # Y_test = keras.utils.to_categorical(Y_test, 10)
    return X_test,Y_test
    
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
       # train_images[:, :, :, i] = (train_images[:, :, :, i] - mean[i]) / std[i]
        test_images[:, :, :, i] = (test_images[:, :, :, i] - mean[i]) / std[i]
    for i in range(train_images.shape[-1]):
        train_images[:, :, :, i] = (train_images[:, :, :, i] - mean[i]) / std[i]
      #  test_images[:, :, :, i] = (test_images[:, :, :, i] - mean[i]) / std[i]
    
    return train_images, test_images

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument(
        "--lsa", "-lsa", help="Likelihood-based Surprise Adequacy", action="store_true"
    )
    parser.add_argument(
        "--dsa", "-dsa", help="Distance-based Surprise Adequacy", action="store_true"
    )
    parser.add_argument(
        "--target",
        "-target",
        help="Target input set (test or adversarial set)",
        type=str,
        default="fgsm",
    )
    parser.add_argument(
        "--save_path", "-save_path", help="Save path", type=str, default="./tmp/"
    )
    parser.add_argument(
        "--batch_size", "-batch_size", help="Batch size", type=int, default=128
    )
    parser.add_argument(
        "--var_threshold",
        "-var_threshold",
        help="Variance threshold",
        type=int,
        default=1e-5,
    )
    parser.add_argument(
        "--upper_bound", "-upper_bound", help="Upper bound", type=int, default=2000
    )
    parser.add_argument(
        "--n_bucket",
        "-n_bucket",
        help="The number of buckets for coverage",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--num_classes",
        "-num_classes",
        help="The number of classes",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--is_classification",
        "-is_classification",
        help="Is classification task",
        type=bool,
        default=True,
    )
    args = parser.parse_args()
    assert args.d in ["mnist", "cifar","combine_cifar","imagenet"], "Dataset should be either 'mnist' or 'cifar'"
    assert args.lsa ^ args.dsa, "Select either 'lsa' or 'dsa'"
    print(args)

    if args.d == "mnist":
       # (x_train, y_train), (x_test, y_test) = mnist.load_data()
        #x_train = x_train.reshape(-1, 28, 28, 1)
        #x_train = x_train.astype('float32') / 255
       # print(x_train[1])
        #x_test = x_test.reshape(-1, 28, 28, 1)
       # ft = np.load('data/mnist_combined_jsma_test_2.npz')
     #   X_test = ft['x_test']
      #  x_test=ft['arr_0']
        x_train=np.load('data/PIE27_x.npy')
        x_train = x_train.reshape(-1, 32, 32, 1)
       # print(x_test[1])
  #    Y_test = ft['y_test']
       # y_test=ft['arr_1']
        y_train=np.load('data/PIE27_y.npy')
        x_test=np.load('data/x_pie9_test.npy')
        x_test = x_test.reshape(-1, 32, 32, 1)
        y_test=np.load('data/y_pie9_test.npy')
        print(y_test)
        y_test = keras.utils.to_categorical(y_test, 68)
        print(y_test)
        # Load pre-trained model.
        model = load_model("data/model_PIE27-9_0.68.h5")
        model.summary()

        # You can select some layers you want to test.
        # layer_names = ["activation_1"]
        # layer_names = ["activation_2"]
        layer_names = ["dense_2"]

        # Load target set.
        x_target = x_test

    elif args.d == "cifar":
        from keras.datasets import cifar100
       # subtract_pixel_mean = True
     
        #(x_train, y_train), (x_test, y_test) = cifar10.load_data()
        #/home/zhangyingyi/adv/
        #ft = np.load('data/cifar10_combined_fgsm_test_vgg16.npz')
     #   X_test = ft['x_test']
       # x_test=ft['arr_0']
   #     x_test=np.load('data/cifar10_test_x.npy')
    #    print(x_test)
     #   y_test=np.load('data/cifar10_test_y.npy')
  #    Y_test = ft['y_test']
        #y_test=ft['arr_1']
      # Normalize data.
     #   x_train = x_train.astype('float32') / 255
      #  x_test = x_test.astype('float32') / 255

    # If subtract pixel mean is enabled
     #   if subtract_pixel_mean:
      #  x_train_mean = np.mean(x_train, axis=0)
       #     print(x_train_mean.shape)
     #   x_train -= x_train_mean
         #   print(x_test.shape)
       # x_test += x_train_mean
        
      #  x_train=X_train
       # x_test=X_test
       # x_test=np.load('data/cifar10_test_x.npy')
        #y_test=np.load('data/cifar10_test_y.npy')
      #  x_test += x_train_mean
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255.
        x_test /= 255.

   # y_train = tf.keras.utils.to_categorical(y_train)
    #y_test = tf.keras.utils.to_categorical(y_test)
        x_test = np.load('/home/youhanmo/202/zyy/adv/cifar10_test_x_full.npy')
        y_test = np.load('/home/youhanmo/202/zyy/adv/cifar10_test_y_full.npy')

    # preprocess data
        #x_train, x_test = pre_processing(x_train, x_test)
        #ft = np.load('/home/youhanmo/zyy/adv/data/cifar100_test_combined_jsma_vgg19.npz')
        #x_test=ft['arr_0']
        #y_test=ft['arr_1']
        y_test = keras.utils.to_categorical(y_test, 10)
        # model = load_model("./model/block1_conv1.h5")
        #cifar10-vgg16_model_alllayers
        model = load_model('/home/youhanmo/202/zyy/adv/cifar10-vgg16_model_alllayers.h5')
        model.summary()

        # layer_names = [
        #     layer.name
        #     for layer in model.layers
        #     if ("activation" in layer.name or "pool" in layer.name)
        #     and "activation_9" not in layer.name
        # ]
       # layer_names = ["activation_6"]
        layer_names = ["dropout_1"]

        # x_target = np.load("./adv/adv_cifar_{}.npy".format(args.target))
        x_target=x_test
        
    elif args.d == "combine_cifar":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        adversarial = 'fgsm'
        x_test,y_test = get_adversarial_cifar10()

        # model = load_model("./model/block1_conv1.h5")
        model = load_model('./model/model_cifar10_resnet20.h5')
        model.summary()

        # layer_names = [
        #     layer.name
        #     for layer in model.layers
        #     if ("activation" in layer.name or "pool" in layer.name)
        #     and "activation_9" not in layer.name
        # ]
        layer_names = ["activation_6"]

        x_target = np.load("./adv/adv_cifar_{}.npy".format(args.target))

    elif args.d == "imagenet":

   #     train_trans = transforms.Compose([
    #        transforms.RandomResizedCrop(224),
     #       transforms.ToTensor()
      #  ])
        # You need to specify ``root`` for ImageNet if you extracted the images into
        # a different folder
    #    train_data = DataLoader(
   #         ImageNet(train=True).transform_first(train_trans),
     #       batch_size=128, shuffle=True)


        # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  #      for x, y in train_data:
        x_train = []
        y_train = []
    #        break
        x_test,y_test = get_imagenet()

        # model = load_model("./model/block1_conv1.h5")
        from keras.applications.vgg19 import VGG19
        model = VGG19(weights='imagenet', include_top=True)
        model.summary()

        # layer_names = [
        #     layer.name
        #     for layer in model.layers
        #     if ("activation" in layer.name or "pool" in layer.name)
        #     and "activation_9" not in layer.name
        # ]
        # layer_names = ["input_1","block1_conv1","block4_conv4 ","block4_pool"]
        layer_names = ["block5_conv4"]
        x_target,_ = get_imagenet()
        # x_target = np.load("./adv/adv_cifar_{}.npy".format(args.target))

    # x_train = x_train.astype("float32")
    # x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
    # x_test = x_test.astype("float32")
    # x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)
  #  from keras.applications.vgg19 import preprocess_input
    #x_train = preprocess_input(x_train)
   # x_test = preprocess_input(x_test)

    if args.lsa:
     #   print(model)
        print(x_train.shape)
        print(x_test.shape)
        test_lsa = fetch_lsa(model, x_train, x_test, "test", layer_names, args)

        # target_lsa = fetch_lsa(model, x_train, x_target, args.target, layer_names, args)
        # target_cov = get_sc(
        #     np.amin(target_lsa), args.upper_bound, args.n_bucket, target_lsa
        # )
        # auc = compute_roc_auc(test_lsa, target_lsa)
        # print(infog("ROC-AUC: " + str(auc * 100)))

        lsa_dict = {}
        for idx in range(len(test_lsa)):
            lsa_dict.update({idx:test_lsa[idx]})
        import pickle
        # file_name = 'combine_lsa_resnet20'
        # file_name = 'lsa_vgg19_random'
        file_name='result/cifar10_vgg16_lsa'
        dictfile = open(file_name + '.dict', 'wb')
        d2 = sorted(lsa_dict.items(), key=lambda x: x[1], reverse=True)
        pickle.dump(d2, dictfile)
        dictfile.close()

    if args.dsa:
        test_dsa = fetch_dsa(model, x_train, x_test, "test", layer_names, args)

        # target_dsa = fetch_dsa(model, x_train, x_target, args.target, layer_names, args)
        # target_cov = get_sc(
        #     np.amin(target_dsa), args.upper_bound, args.n_bucket, target_dsa
        # )
        #
        # auc = compute_roc_auc(test_dsa, target_dsa)
        # print(infog("ROC-AUC: " + str(auc * 100)))

        dsa_dict = {}
        for idx in range(len(test_dsa)):
            dsa_dict.update({idx:test_dsa[idx]})
        import pickle
        # file_name = 'combine_dsa_resnet20'
        # file_name = 'dsa_vgg19_random'
        file_name='result/cifar10_vgg16_dsa'
        dictfile = open(file_name + '.dict', 'wb')
        d2 = sorted(dsa_dict.items(), key=lambda x: x[1], reverse=True)
        pickle.dump(d2, dictfile)
        dictfile.close()

    # print(infog("{} coverage: ".format(args.target) + str(target_cov)))
