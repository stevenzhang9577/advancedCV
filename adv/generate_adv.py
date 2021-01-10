#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import keras
from keras import Model,Input
from keras.models import load_model
from keras.layers import Activation,Flatten
import math
import numpy as np
import pandas as pd
import foolbox
from tqdm import tqdm
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.datasets import cifar100
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import scipy
import sys,os
# import SVNH_DatasetUtil
import itertools
sys.path.append('./fashion-mnist/utils')
# import mnist_reader
import warnings
warnings.filterwarnings("ignore")
import multiprocessing
import datetime

from keras.applications import inception_v3,vgg19,resnet50
import traceback
import argparse


def adv_func(x,y,model_path='./model/model_mnist.hdf5',dataset='mnist',attack='fgsm',mean=0):
    keras.backend.set_learning_phase(0)
    model=load_model(model_path)

    # model = inception_v3.InceptionV3(weights='imagenet')
  #  model = vgg19.VGG19(weights='imagenet')
   # preprocessing = (np.array([104, 116, 123]), 1)
    foolmodel=foolbox.models.KerasModel(model,bounds=(-5,5),preprocessing=(mean,1))
    #foolmodel = foolbox.models.KerasModel(model, bounds=(0, 255), preprocessing= preprocessing)
    if attack=='cw':
        #attack=foolbox.attacks.IterativeGradientAttack(foolmodel)
        attack=foolbox.attacks.CarliniWagnerL2Attack(foolmodel)
    elif attack=='fgsm':
        # FGSM
        attack=foolbox.attacks.GradientSignAttack(foolmodel)
    elif attack=='bim':
        # BIM
        metric = foolbox.distances.MAE
        attack=foolbox.attacks.L1BasicIterativeAttack(foolmodel)
    elif attack=='jsma':
        # JSMA
        attack=foolbox.attacks.SaliencyMapAttack(foolmodel)
        # CW
        #attack=foolbox.attacks.DeepFoolL2Attack(foolmodel)
    result=[]
    if dataset=='mnist':
        w,h=28,28
    elif dataset=='cifar10' or dataset=='cifar100':
        w,h=32,32
    elif dataset=='imagenet':
        w,h=224,224
    else:
        return False

    y_list = []
    for b in range(x.shape[0]):
        y_list.append(y)
    print(x.shape)
    y_list = np.array(y_list)
    print(y_list.shape)
    print(len(x))
    print(len(y_list))
    es=[0.01,0.02,0.03,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
    adv = attack(x, y_list,theta=2)
    confidence =20
    # print("adv", adv)

    # for image in tqdm(x):
    #     # print("ok1")
    #     try:
    #         print("ok1")
    #         print(y)
    #
    #
    #
    #         adv = attack(image.reshape(w, h, -1), y_list)
    #         print("adv", adv)
    #         #adv=attack(image.reshape(28,28,-1),label=y,steps=1000,subsample=10)
    #         #adv=attack(image.reshape(w,h,-1),y,epsilons=[0.01,0.1],steps=10)
    #         if attack!='fgsm':
    #             adv=attack(image.reshape(w,h,-1),y)
    #             adv=attack(image.reshape(w,h,-1),y)
    #             adv=attack(image.reshape(w,h,-1),y)
    #             print("adv",adv)
    #         else:
    #             adv=attack(image.reshape(w,h,-1),y,[0.01,0.1])
    #             print("adv", adv)
    #
    #         if isinstance(adv,np.ndarray):
    #             print(adv)
    #             print("ok")
    #             result.append(adv)
    #         else:
    #             print('adv fail')
    #     except:
    #         traceback.print_exc()
    #         print("fail")
    #         pass
    # print(result)
    return adv


def generate_mnist_sample(label,attack):
    #(X_train, Y_train), (X_test, Y_test) = mnist.load_data()  # 28*28
   # X_test=np.load('new_adv/mnist_validation_x.npy')
    #Y_test=np.load('new_adv/mnist_validation_y.npy')
    X_test=np.load('new_adv/mnist_test_x.npy')
    Y_test=np.load('new_adv/mnist_test_y.npy')
    
  #  X_train = X_train.astype('float32').reshape(-1,28,28,1)
    X_test = X_test.astype('float32').reshape(-1,28,28,1)
  #  X_train /= 255
  #  X_test /= 255
   # print(X_test)
    image_org=X_test[Y_test==label]
    adv=adv_func(image_org,label,model_path='LeNet-5.h5',dataset='mnist',attack=attack)
    return adv

def generate_cifar_sample(label,attack):
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()  # 32*32
    #ft=np.load('/home/youhanmo/202/data/cifar10_train.npz')
        #print(ft.files)
    #X_train=ft['x']
    #Y_train=ft['y']
    X_train = X_train.astype('float32').reshape(-1,32,32,3)
    #X_test = X_test.astype('float32').reshape(-1,32,32,3)
    X_test=np.load('cifar10_test_x_full.npy')
    Y_test=np.load('cifar10_test_y_full.npy')
    
    X_train /= 255.0
    #X_test /= 255.0
    print('X_train')
    print(X_train)
    print('X_test')
    print(X_test)
    #x_train_mean = np.mean(X_train, axis=0)
    Y_train=Y_train.reshape(-1)
    Y_test=Y_test.reshape(-1)

    image_org=X_test[Y_test==label]

    adv=adv_func(image_org,label,model_path='/home/youhanmo/202/lsa/Sat-Nov-14-09-42-21-2020.model.060-0.8503.hdf5',dataset='cifar10',attack=attack,mean=0)
    return adv

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
    
def generate_cifar100_sample(label,attack):
    # (X_train, Y_train), (X_test, Y_test) = cifar100.load_data(label_mode='coarse')  # 32*32
    
    # print(X_test.shape)
    # print(Y_test.shape)
    X_test,Y_test=get_cifar_gen()
    print(X_test.shape)
    print(Y_test)
    Y_test=Y_test.reshape(-1)
    image_org=X_test[Y_test==label]
    print(image_org)
#/home/youhanmo/zyy/vgg19_dense.h5
    adv=adv_func(image_org,label,model_path='/home/youhanmo/zyy/vgg19_dense.h5',dataset='cifar100',attack=attack)
    print(adv)
    return adv

def generate_fashion_sample(label,attack):
    path='./fashion-mnist/data/fashion'
    X_train, Y_train = mnist_reader.load_mnist(path, kind='train')
    X_test, Y_test = mnist_reader.load_mnist(path, kind='t10k')
    X_train = X_train.astype('float32').reshape(-1,28,28,1)
    X_test = X_test.astype('float32').reshape(-1,28,28,1)
    X_train /= 255
    X_test /= 255

    image_org=X_test[Y_test==label]
    adv=adv_func(image_org,label,model_path='./model/model_fashion.hdf5',dataset='mnist',attack=attack)
    return adv

def generate_svhn_sample(label,attack):

    (X_train, Y_train), (X_test, Y_test) = SVNH_DatasetUtil.load_data()  # 32*32

    image_org=X_test[np.argmax(Y_test,axis=1)==label]

    adv=adv_func(image_org,label,model_path='./model/model_svhn.hdf5',dataset='cifar10',attack=attack)
    return adv

def generate_imagenet_sample(X_test, Y_test, label,attack):
    # data_path = './data/imagenet.npz'
    # data = np.load(data_path)
    # X_test, Y_test = data['x_test'], data['y_test']
    # exp_id = kwargs['exp_id']
    # if exp_id == 'vgg19':

    # X_test = X_test.astype('float32').reshape(-1, 224, 224, 3)
    # X_train /= 255
    # X_test /= 255
    # X_test = vgg19.preprocess_input(X_test,mode='torch')
    # print(X_test[0])
    # Y_test = keras.utils.to_categorical(Y_test, num_classes=1000)
    # if exp_id == 'resnet50':
    # X_test = resnet50.preprocess_input(X_test)
    # Y_test = keras.utils.to_categorical(Y_test, num_classes=1000)
    # X_test = inception_v3.preprocess_input(X_test)
    # Y_test = keras.utils.to_categorical(Y_test, num_classes=1000)
    # print(Y_test)
    # list_temp = list(set(Y_test))
    # print(sorted(list(set(Y_test))))
    # list_temp = [int(x) for x in list_temp]
    # list_temp = sorted(list_temp)
    # print(list_temp)
    # print(len(list_temp))

    # for i in range(len(list_temp)):
    #     if i+1 == len(list_temp):
    #         break
    #     if list_temp[i+1] - list_temp[i] != 1:
    #         print(list_temp[i])
    print(Y_test[0])
    print(X_test.shape)
    print(Y_test.shape)


    image_org = X_test[Y_test == str(label)]
    print(image_org.shape)
    if image_org.shape[0]==0:
        return []
    print("image_org", image_org.shape)
    adv = adv_func(image_org, label, model_path='./model/model_fashion.hdf5', dataset='imagenet', attack=attack)
    return adv



def generate_adv_sample(dataset,attack):
    if dataset=='mnist':
        sample_func=generate_mnist_sample
    elif dataset=='svhn':
        sample_func=generate_svhn_sample
    elif dataset=='fashion':
        sample_func=generate_fashion_sample
    elif dataset=='cifar10':
        sample_func=generate_cifar_sample
    elif dataset=='cifar100' or dataset=='cifar20':
        sample_func=generate_cifar100_sample
    elif dataset=='imagenet':
        data_path = './data/imagenet-val-5000.npz'
        data = np.load(data_path)
        X_test, Y_test = data['x_test'], data['y_test']
        sample_func=generate_imagenet_sample
    else:
        print('erro')
        return
    image=[]
    label=[]
    for i in range(10):
        print(i)
        start = datetime.datetime.now()
        adv=sample_func(label=i,attack=attack)
        print(adv)
        if len(adv)==0:
            continue
        # print(adv)
        temp_image=adv
        temp_label=i*np.ones(len(adv))
        image.append(temp_image.copy())
        label.append(temp_label.copy())
        elapsed = (datetime.datetime.now() - start)
        print("Time used: ", elapsed)
    image=np.concatenate(image,axis=0)
    label=np.concatenate(label,axis=0)
    # print(label)
    np.save('./test/{}_{}_image_vgg16_0.85'.format(attack,dataset),image)
    np.save('./test/{}_{}_label_vgg16_0.85'.format(attack,dataset),label)

    # np.save('./imagenet_adv/{}_{}_image_0'.format(attack, dataset), image)
    # np.save('./imagenet_adv/{}_{}_label_0'.format(attack, dataset), label)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", "-count", help="", type=int)
    args = parser.parse_args()

    '''
    mnist svhn fashion cifar10 cifar20
    cw fgsm bim jsma
    '''
    # data_lst=['svhn','fashion','cifar10','mnist']
    # attack_lst=['cw','fgsm','bim','jsma']
    # pool = multiprocessing.Pool(processes=8)
    # for dataset,attack in (itertools.product(data_lst,attack_lst)):
    #     pool.apply_async(generate_adv_sample, (dataset,attack))
    # pool.close()
    # pool.join()

    # generate_adv_sample('cifar10','bim')
    # generate_adv_sample('cifar10', 'fgsm')
    # generate_adv_sample('imagenet', 'cw')

    # generate_adv_sample('mnist', 'fgsm')
    # generate_adv_sample('mnist', 'bim')
    # generate_adv_sample('mnist', 'jsma')
    #
    # generate_adv_sample('cifar10', 'fgsm')
    # generate_adv_sample('cifar10', 'bim')
    generate_adv_sample('cifar10', 'jsma')

