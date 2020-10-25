# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 11:01:07 2020

@author: DELL
"""

from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import add
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import GlobalAveragePooling2D, Lambda
from numpy import argmax
from tensorflow.keras import backend
import os
import base64
import glob
import cv2
import numpy as np
from numba import cuda

from .fetch_colour import detect_colour


def cnn_classification(request):

    imgpath_base64 = request.data['imgFile']

    imgdata = base64.b64decode(imgpath_base64)
    imgpath = 'test.jpg'
    with open(imgpath, 'wb') as f:
        f.write(imgdata)

    imgrows, imgclms, channel, num_classes = 256, 256, 3, 6

    def createModel_1():
        imgrows, imgclms, channel, num_classes = 256, 256, 3, 6
        optmz = optimizers.RMSprop(lr=0.0001)
        inputs = Input(shape=(imgrows, imgclms, channel))
        x = Conv2D(32, (3, 3), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.25)(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.25)(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.25)(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.25)(x)
        x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=x)

        model.compile(loss='categorical_crossentropy',
                      optimizer=optmz,
                      metrics=['accuracy'])

        return model

        # Step 4

    model = createModel_1()  # This is meant for training

    def createModel_2():
        imgrows, imgclms, channel, num_classes = 256, 256, 3, 6
        optmz = optimizers.RMSprop(lr=0.0001)
        inputs = Input(shape=(imgrows, imgclms, channel))
        x = Conv2D(32, (3, 3), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.25)(x)
        y = x
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = add([x, y])
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = Dropout(0.25)(x)
        y = x
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = add([x, y])
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.25)(x)
        x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=x)

        model.compile(loss='categorical_crossentropy',
                      optimizer=optmz,
                      metrics=['accuracy'])

        return model

    model_1 = createModel_2()

    def resLyr(inputs,
               numFilters=16,
               kernelSz=3,
               strides=1,
               activation='relu',
               batchNorm=True,
               convFirst=True,
               lyrName=None):
        convLyr = Conv2D(numFilters,
                         kernel_size=kernelSz,
                         strides=strides,
                         padding='same',
                         kernel_initializer='he_normal',
                         kernel_regularizer=l2(1e-4),
                         name=lyrName + '_conv' if lyrName else None)
        x = inputs

        if convFirst:
            x = convLyr(x)

            if batchNorm:
                x = BatchNormalization(name=lyrName + '_bn' if lyrName else None)(x)

            if activation is not None:
                x = Activation(activation,
                               name=lyrName + '_' + activation if lyrName else None)(x)
        else:
            if batchNorm:
                x = BatchNormalization(name=lyrName + '_bn' if lyrName else None)(x)

            if activation is not None:
                x = Activation(activation,
                               name=lyrName + '_' + activation if lyrName else None)(x)

            x = convLyr(x)
        return x

        # Step 4

    def resBlkV1(inputs,
                 numFilters=16,
                 numBlocks=3,
                 downsampleOnFirst=True,
                 names=None):
        x = inputs

        for run in range(0, numBlocks):
            strides = 1
            blkStr = str(run + 1)

            if downsampleOnFirst and run == 0:
                strides = 2

            y = resLyr(inputs=x,
                       numFilters=numFilters,
                       strides=strides,
                       lyrName=names + '_Blk' + blkStr + '_Res1' if names else None)
            y = resLyr(inputs=y,
                       numFilters=numFilters,
                       activation=None,
                       lyrName=names + '_Blk' + blkStr + '_Res2' if names else None)

            if downsampleOnFirst and run == 0:
                x = resLyr(inputs=x,
                           numFilters=numFilters,
                           kernelSz=1,
                           strides=strides,
                           activation=None,
                           batchNorm=False,
                           lyrName=names + '_Blk' + blkStr + '_lin' if names else None)

            x = add([x, y],
                    name=names + '_Blk' + blkStr + '_add' if names else None)
            x = Activation('relu',
                           name=names + '_Blk' + blkStr + '_relu' if names else None)(x)

        return x

        # Step 5

    def createResNetV1(inputShape=(imgrows, imgclms, channel), numClasses=num_classes):
        inputs = Input(shape=inputShape)
        v = resLyr(inputs, lyrName='Inpt')
        v = resBlkV1(inputs=v, numFilters=16, numBlocks=3, downsampleOnFirst=False, names='Stg1')
        v = resBlkV1(inputs=v, numFilters=32, numBlocks=3, downsampleOnFirst=True, names='Stg2')
        v = resBlkV1(inputs=v, numFilters=64, numBlocks=3, downsampleOnFirst=True, names='Stg4')
        v = AveragePooling2D(pool_size=4, name='AvgPool')(v)
        v = Flatten()(v)
        outputs = Dense(numClasses, activation='softmax', kernel_initializer='he_normal')(v)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model

        # Step 6

    model_2 = createResNetV1()

    # ###Convolutional Block
    def conv2d(x, numfilt, filtsz, strides=1, pad='same', act=True, name=None):
        x = Conv2D(numfilt, filtsz, strides, padding=pad, data_format='channels_last', use_bias=False,
                   name=name + 'conv2d')(x)
        x = BatchNormalization(axis=3, scale=False, name=name + 'conv2d' + 'bn')(x)
        if act:
            x = Activation('relu', name=name + 'conv2d' + 'act')(x)
        return x

    # ##Inception ResNet A block
    def incresA(x, scale, name=None):
        pad = 'same'
        branch0 = conv2d(x, 32, 1, 1, pad, True, name=name + 'b0')
        branch1 = conv2d(x, 32, 1, 1, pad, True, name=name + 'b1_1')
        branch1 = conv2d(branch1, 32, 3, 1, pad, True, name=name + 'b1_2')
        branch2 = conv2d(x, 32, 1, 1, pad, True, name=name + 'b2_1')
        branch2 = conv2d(branch2, 48, 3, 1, pad, True, name=name + 'b2_2')
        branch2 = conv2d(branch2, 64, 3, 1, pad, True, name=name + 'b2_3')
        branches = [branch0, branch1, branch2]
        mixed = Concatenate(axis=3, name=name + '_concat')(branches)
        filt_exp_1x1 = conv2d(mixed, 384, 1, 1, pad, False, name=name + 'filt_exp_1x1')
        final_lay = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                           output_shape=backend.int_shape(x)[1:],
                           arguments={'scale': scale},
                           name=name + 'act_scaling')([x, filt_exp_1x1])
        return final_lay

    ##Stem block
    img_input = Input(shape=(imgrows, imgclms, channel))

    x = conv2d(img_input, 32, 3, 2, 'valid', True, name='conv1')
    x = conv2d(x, 32, 3, 1, 'valid', True, name='conv2')
    x = conv2d(x, 64, 3, 1, 'valid', True, name='conv3')

    x_11 = MaxPooling2D(3, strides=1, padding='valid', name='stem_br_11' + '_maxpool_1')(x)
    x_12 = conv2d(x, 64, 3, 1, 'valid', True, name='stem_br_12')

    x = Concatenate(axis=3, name='stem_concat_1')([x_11, x_12])

    x_21 = conv2d(x, 64, 1, 1, 'same', True, name='stem_br_211')
    x_21 = conv2d(x_21, 64, [1, 7], 1, 'same', True, name='stem_br_212')
    x_21 = conv2d(x_21, 64, [7, 1], 1, 'same', True, name='stem_br_213')
    x_21 = conv2d(x_21, 96, 3, 1, 'valid', True, name='stem_br_214')

    x_22 = conv2d(x, 64, 1, 1, 'same', True, name='stem_br_221')
    x_22 = conv2d(x_22, 96, 3, 1, 'valid', True, name='stem_br_222')

    x = Concatenate(axis=3, name='stem_concat_2')([x_21, x_22])

    x_31 = conv2d(x, 192, 3, 1, 'valid', True, name='stem_br_31')
    x_32 = MaxPooling2D(3, strides=1, padding='valid', name='stem_br_32' + '_maxpool_2')(x)
    x = Concatenate(axis=3, name='stem_concat_3')([x_31, x_32])

    # ##Inception-ResNet Network
    x = incresA(x, 0.15, name='incresA_1')

    # 35 × 35 to 17 × 17 reduction module.
    x_red_11 = MaxPooling2D(3, strides=2, padding='valid', name='red_maxpool_1')(x)
    x_red_12 = conv2d(x, 384, 3, 2, 'valid', True, name='x_red1_c1')
    x_red_13 = conv2d(x, 256, 1, 1, 'same', True, name='x_red1_c2_1')
    x_red_13 = conv2d(x_red_13, 256, 3, 1, 'same', True, name='x_red1_c2_2')
    x_red_13 = conv2d(x_red_13, 384, 3, 2, 'valid', True, name='x_red1_c2_3')

    x = Concatenate(axis=3, name='red_concat_1')([x_red_11, x_red_12, x_red_13])

    # 17 × 17 to 8 × 8 reduction module.
    x_red_21 = MaxPooling2D(3, strides=2, padding='valid', name='red_maxpool_2')(x)

    x_red_22 = conv2d(x, 256, 1, 1, 'same', True, name='x_red2_c11')
    x_red_22 = conv2d(x_red_22, 384, 3, 2, 'valid', True, name='x_red2_c12')

    x_red_23 = conv2d(x, 256, 1, 1, 'same', True, name='x_red2_c21')
    x_red_23 = conv2d(x_red_23, 256, 3, 2, 'valid', True, name='x_red2_c22')

    x_red_24 = conv2d(x, 256, 1, 1, 'same', True, name='x_red2_c31')
    x_red_24 = conv2d(x_red_24, 256, 3, 1, 'same', True, name='x_red2_c32')
    x_red_24 = conv2d(x_red_24, 256, 3, 2, 'valid', True, name='x_red2_c33')

    x = Concatenate(axis=3, name='red_concat_2')([x_red_21, x_red_22, x_red_23, x_red_24])

    # TOP
    x = GlobalAveragePooling2D(data_format='channels_last')(x)
    x = Dropout(0.6)(x)
    x = Dense(num_classes, activation='softmax')(x)

    model_3 = Model(img_input, x, name='inception_resnet_v2')

    def createModel_3():
        imgrows, imgclms, channel, num_classes = 256, 256, 3, 6
        optmz = optimizers.RMSprop(lr=0.0001)
        inputs = Input(shape=(imgrows, imgclms, channel))
        x = Conv2D(32, (3, 3), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.25)(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.25)(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.25)(x)
        x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=x)

        model.compile(loss='categorical_crossentropy',
                      optimizer=optmz,
                      metrics=['accuracy'])

        return model

    model_4 = createModel_3()

    def createModel_sleevelength():
        imgrows, imgclms, channel, num_classes = 256, 256, 3, 4
        optmz = optimizers.RMSprop(lr=0.0001)
        inputs = Input(shape=(imgrows, imgclms, channel))
        x = Conv2D(32, (3, 3), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.25)(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.25)(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.25)(x)
        x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=x)

        model.compile(loss='categorical_crossentropy',
                      optimizer=optmz,
                      metrics=['accuracy'])

        return model

    def createModel_length():
        imgrows, imgclms, channel, num_classes = 256, 256, 3, 5
        optmz = optimizers.RMSprop(lr=0.0001)
        inputs = Input(shape=(imgrows, imgclms, channel))
        x = Conv2D(32, (3, 3), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.25)(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.25)(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.25)(x)
        x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=x)

        model.compile(loss='categorical_crossentropy',
                      optimizer=optmz,
                      metrics=['accuracy'])

        return model

    def define_stacked_model(members):
        for i in range(len(members)):
            model = members[i]
            for layer in model.layers:
                # make not trainable
                layer.trainable = False
        # define multi-headed input
        ensemble_visible = [model.input for model in members]
        # concatenate merge output from each model
        ensemble_outputs = [model.output for model in members]
        merge = Concatenate()(ensemble_outputs)
        hidden = Dense(10, activation='relu')(merge)
        output = Dense(num_classes, activation='softmax')(hidden)
        model = Model(inputs=ensemble_visible, outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    # fit a stacked model
    # def fit_stacked_model(model, inputX, inputy):
    #     # prepare input data
    #     X = [inputX for _ in range(len(model.input))]
    #     # encode output data
    #     inputy_enc = inputy
    #     # fit model
    #     history = model.fit(X, inputy_enc, epochs=20, verbose=1)
    #     return history

    # ## make a prediction with a stacked model
    def predict_stacked_model(model, inputX):
        X = [inputX for _ in range(len(model.input))]
        return model.predict(X, verbose=0)

    all_models = list()
    all_models.append(model)
    all_models.append(model_1)
    all_models.append(model_2)
    all_models.append(model_3)
    all_models.append(model_4)

    members = all_models
    print('Loaded %d models' % len(members))

    # define ensemble model
    Classification_model = define_stacked_model(members)
    filepath_pattern = os.getcwd() + r'\\GarmentClassification\\deployment\\weights\\pattern_pred_conv.hdf5'
    Classification_model.load_weights(filepath_pattern)
    shape = (256, 256)
    labelname = ['floral', 'lace', 'polkadots', 'print', 'stripes', 'unicolors']
    images = []
    for img in glob.glob(imgpath):
        n = cv2.imread(img)
        n = cv2.resize(n, shape)
        images.append(n)
    images = np.array(images)
    yhat = predict_stacked_model(Classification_model, images)
    pattern = [labelname[i] for i in argmax(yhat, axis=1)]
    print(pattern)

    Classification_model_sleeve_length = createModel_sleevelength()
    filepath_sleeve_length = os.getcwd() + r'\\GarmentClassification\\deployment\\weights\\sleeve_length_pred_conv_5.hdf5'
    Classification_model_sleeve_length.load_weights(filepath_sleeve_length)
    shape = (256, 256)
    sleeve_labelname = ['half', 'long', 'short', 'sleeveless']
    images = []
    for img in glob.glob(imgpath):
        n = cv2.imread(img)
        n = cv2.resize(n, shape)
        images.append(n)
    images = np.array(images)
    yhat = Classification_model_sleeve_length.predict(images)
    sleeve_label = [sleeve_labelname[i] for i in argmax(yhat, axis=1)]
    print(sleeve_label)

    Classification_model_length = createModel_length()
    filepath_length = os.getcwd() + r'\\GarmentClassification\\deployment\\weights\\length_pred_conv_n.hdf5'
    Classification_model_length.load_weights(filepath_length)
    shape = (256, 256)
    length_labelname = ['3-4', 'knee', 'long', 'normal', 'short']
    images = []
    for img in glob.glob(imgpath):
        n = cv2.imread(img)
        n = cv2.resize(n, shape)
        images.append(n)
    images = np.array(images)
    yhat = Classification_model_length.predict(images)
    length_label = [length_labelname[i] for i in argmax(yhat, axis=1)]
    print(length_label)

    detected_colour = detect_colour(imgpath)

    # ### clearing up CUDA GPU
    device = cuda.get_current_device()
    device.reset()
    prediction_dictionary = {'pattern': pattern[0], 'sleeve_length': sleeve_label[0], 'length': length_label[0]}
    prediction_dictionary.update(detected_colour)
    os.remove(imgpath)
    return prediction_dictionary
