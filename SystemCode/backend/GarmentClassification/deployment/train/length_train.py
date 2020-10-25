import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import cv2
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers

import sklearn.metrics as metrics
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv(os.getcwd()+r'/FashionSetGeneration/images/fashion/data_dresses.csv', sep=';')

df = df.dropna(axis=0, subset=['length'])
df = df.reset_index(drop=True)

images = []
img_labels = []
for i in range(int(len(df))):
    img = cv2.imread(os.getcwd() + r'\FashionSetGeneration\images\fashion\\' + df['img_path'][i])
    images.append(img)
    img_labels.append(df['length'][i])

labels = pd.get_dummies(img_labels).values
labelname = pd.get_dummies(img_labels).columns

dfc = pd.DataFrame({'key': np.array(img_labels),
                    'data': range(len(img_labels))}, columns=['key', 'data'])
dfc.groupby('key').count()

trlen = int(len(images) * 0.75)
tslen = len(images) - trlen
print(trlen, tslen)

trDat = images[:trlen]
trLbl = labels[:trlen]

tsDat = images[trlen:]
tsLbl = labels[trlen:]

trDat = np.array(trDat)
tsDat = np.array(tsDat)

imgrows = trDat[0].shape[0]
imgclms = trDat[0].shape[1]
channel = trDat[0].shape[2]
num_classes = len(np.unique(img_labels))

optmz = optimizers.RMSprop(lr=0.0001)  # Step 1


# Step 3
def createModel():
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

    # Step 4


model = createModel()  # This is meant for training
modelGo = createModel()  # This is used for final testing

model.summary()  # Step 5

modelname = '\length_pred_conv_n'  # Step 1
folderpath = os.getcwd()
filepath = folderpath + modelname + ".hdf5"
checkpoint = ModelCheckpoint(filepath,
                             monitor='val_accuracy',
                             verbose=0,
                             save_best_only=True,
                             mode='max')
early = EarlyStopping(monitor="val_loss",
                      mode="min",
                      patience=8, restore_best_weights=True)

history = model.fit(trDat,  # Training data
                    trLbl,  # Training label
                    validation_data=(tsDat, tsLbl),  # Validation data and label
                    epochs=8,  # The amount of epochs to be trained
                    batch_size=20,
                    shuffle=True  # To shuffle the training data
                    )

# Visualizing the training performance
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='val_Loss')
plt.legend()
plt.title('Loss evolution')
# plt.savefig(os.getcwd() + '\GarmentClassification\deployment\length_train_loss.jpg')

plt.subplot(2, 2, 2)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.title('Accuracy evolution')
# plt.savefig(os.getcwd() + '\GarmentClassification\deployment\length_train_accuracy.jpg')

model.save_weights(filepath)
modelGo.load_weights(filepath)
modelGo.compile(loss='categorical_crossentropy',
                optimizer=optmz,
                metrics=['accuracy'])

predicts = modelGo.predict(tsDat)  # Step 2
print("Prediction completes.")

predout = np.argmax(predicts, axis=1)
testout = np.argmax(tsLbl, axis=1)

testScores = metrics.accuracy_score(testout, predout)  # Step 3

# Step 4
print("Best accuracy (on testing dataset): %.2f%%" % (testScores * 100))
# print(metrics.classification_report(testout,
#                                     predout,
#                                     target_names=labelname,
#                                     digits=5))

confusion = metrics.confusion_matrix(testout, predout)
print(confusion)

model.save(filepath)

