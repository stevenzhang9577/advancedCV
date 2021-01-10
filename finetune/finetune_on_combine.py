import keras
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import os
from keras.utils.np_utils import to_categorical
import argparse
from feature_utils import get_test_csv,get_validate_csv,get_model,get_state_number,get_data,data_proprecessing
from keras.preprocessing.image import ImageDataGenerator
model_name = 'cifar10_vgg16'
data_augmention = False
visible_gpu = 1
batch_size = 32
#import tensorflow as tf
import tensorflow.compat.v1 as tf
import random


seed_value = 9527
tf.set_random_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

# parser = argparse.ArgumentParser()
# parser.add_argument('--model', type=str, help='model name', default=model_name)
# parser.add_argument('--gpu', type=str, help='visible devices', default=visible_gpu)
# parser.add_argument('--batch_size', type=int, help='batch size for dataloader', default=batch_size)
# parser.add_argument('--epochs', type=int, help='num epochs', default=epochs)
# parser.add_argument('--lr', type=float, help='initial learning rate', default=0.1)
# parser.add_argument('--checkpoint', type=str, help='train from a previous model', default=None)
# parser.add_argument('--early_stop', type=bool, help='apply early stop', default=False)
# parser.add_argument('--optimizer', type=str, help='choose optimizer', default='sgd')
# parser.add_argument('--finetune', type=str, help='choose retrain set', default='DSA')
# args = parser.parse_args()





def lr_schedule(epoch):
    lr = 1e-5
    #if epoch > 180:
    #    lr *= 0.5e-3
    #elif epoch > 160:
    #    lr *= 1e-3
    #elif epoch > 120:
    #    lr *= 1e-2
    #elif epoch > 80:
    #    lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

prioritize_type = 'combine_pos_id3901_1.9_50'

model_type = model_name
save_dir = os.path.join(os.getcwd(),'finetune' ,'cifar10_vgg16_'+prioritize_type+'_r0.85_p0.1')
model_name = '%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=False)

lr_scheduler = LearningRateScheduler(lr_schedule)
callbacks = [checkpoint,lr_scheduler]


if __name__ == '__main__':
    origin_model = keras.models.load_model('Sat-Nov-14-09-42-21-2020.model.060-0.8503.hdf5')
    origin_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #x_train = np.concatenate((x_train, x_test_100), axis=0)
    #y_train = np.concatenate((y_train, y_test_100), axis=0)
    # x_train -= X_train_mean
    exp_id = model_type
    x_sample = np.load('adv/combine_pos_adv_id3901_50_1.9.npz')['x']
    y_sample = np.load('adv/combine_pos_adv_id3901_50_1.9.npz')['y']
    # x_sample = np.load('x_error.npy')[0:500]
    # y_sample = np.load('y_error.npy')[0:500]
    x_train = x_sample
    y_train = y_sample
    # x_train = x_train.astype('float32') / 255
    x_train = data_proprecessing(exp_id)(x_train)
    y_train = to_categorical(y_train,10)

    x_test, y_test = get_data(exp_id)()
    x_test = data_proprecessing(exp_id)(x_test)
    y_test = to_categorical(y_test,10)
    # scores = origin_model.evaluate(x_test, y_test)
    # print("Baseline Accuracy: %.2f%%" % (scores[1] * 100))
    x_validation = np.load('cifar10_validation_x_full.npy')
    y_validation = np.load('cifar10_validation_y_full.npy')
    # x_validation = x_validation.astype('float32') / 255
    x_validation = data_proprecessing(exp_id)(x_validation)
    y_validation = to_categorical(y_validation)


    scores = origin_model.evaluate(x_validation,y_validation,verbose=2)
    print("Now Vali Accuracy: %.2f%%" % (scores[1] * 100))

    print(x_train.shape)
    if not data_augmention:
        origin_model.fit(x_train, y_train, epochs=5, batch_size=32,callbacks=callbacks,\
                         validation_data=(x_validation, y_validation),verbose=2)
    else:
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)
        origin_model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                            validation_data=(x_validation, y_validation),
                            epochs=10, verbose=2, workers=4,
                            callbacks=callbacks)


    scores = origin_model.evaluate(x_test,y_test,verbose=2)
    print("New Test Accuracy: %.2f%%" % (scores[1] * 100))
    # print("Now Error: %.2f%%" % (100 - scores[1] * 100))
    # model.fit(x_train, y_train,
    #           batch_size=batch_size,
    #           epochs=epochs,
    #           validation_data=(x_test, y_test),
    #           shuffle=True,
    #           callbacks=callbacks)