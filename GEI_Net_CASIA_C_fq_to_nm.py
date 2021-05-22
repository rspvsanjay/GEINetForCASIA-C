import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import cv2
import os
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.layers.normalization import BatchNormalization
import csv
from tensorflow.keras import optimizers


src_dir = '/content/drive/My Drive/Sanjay/CASIA_C_Silhouette_centered_Alinged_Energy_Image/'
train_imgs = []
train_labels = []
test_imgs = []
test_labels = []
subjects = os.listdir(src_dir)
numberOfSubject = len(subjects)
print('Number of Subjects: ', numberOfSubject)
for i in range(1, numberOfSubject + 1):  # numberOfSubject
    path2 = (src_dir + subjects[i - 1] + '/')
    sequences = os.listdir(path2)
    numberOfsequences = len(sequences)
    print('Subject Index: ',i)
    for j in range(0, 4):
        path3 = path2 + 'fn0' + str(j) + '.png'
        print(path3 + ' training data')
        img = Image.open(path3)
        img = img.resize((200, 200))
        img = img.crop((45, 0, 145, 199))
        img = img.resize((80, 128))
    #img = cv2.imread(path3 , 0)
        x3d = img_to_array(img)
        x = np.expand_dims(x3d[:, :, 0], axis=2)
        train_imgs.append(x)
        label = [0] * numberOfSubject
        label[i-1] = 1
        train_labels.append(label)

    for j in range(0, 2):
        path3 = path2 + 'fq0' + str(j) + '.png'
        print(path3 + ' testing data')
        img = Image.open(path3)
        img = img.resize((200, 200))
        img = img.crop((45, 0, 145, 199))
        img = img.resize((80, 128))
    #img = cv2.imread(path3 , 0)
        x3d = img_to_array(img)
        x = np.expand_dims(x3d[:, :, 0], axis=2)
        test_imgs.append(x)
        label = [0] * numberOfSubject
        label[i-1] = 1
        test_labels.append(label)

x_train = np.array(train_imgs)
y_train = np.array(train_labels)

x_test = np.array(test_imgs)
y_test = np.array(test_labels)

save_dir = '/content/drive/My Drive/GEINet_and_PEINet/GEI_Net_Models/'
model_name = 'keras_GEINet_casia_c_fq2nm.h5'
print(src_dir, 'src_dir')
batch_size = 4
num_classes = numberOfSubject
epochs = 80
#178, 256, 1
model = Sequential()

model.add(Conv2D(filters=18, input_shape=(128, 80, 1), kernel_size=(7, 7), strides=1, activation='relu', padding='valid'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=45, kernel_size=(5, 5), strides=1, activation='relu', padding='valid'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
model.add(BatchNormalization())
model.add(Flatten())

model.add(Dense(1024))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()
#opt = keras.optimizers.rmsprop(lr=0.001)
#opt = tf.keras.optimizers.RMSprop(learning_rate=0.001)
opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#x_train, y_train, x_test, y_test = load_90_degree_gei_for_experiment1(src_dir, 'st2nm')

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
scores = model.evaluate(x_test, y_test, verbose=1)
predict1 = model.predict(x_test, batch_size=batch_size, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
# print('Test y_test :', y_test[5])
print('nm')
xx, yy = predict1.shape
print('xx :', xx)
print('yy :', yy)



accuracy1 = []
for num1 in range(0, 20):
    count = 0
    predict1 = model.predict(x_test, batch_size=batch_size, verbose=0)
    print('rank : ', num1 + 1)
    for num2 in range(0, xx):
        matrix1 = []
        arr = predict1[num2]
        for num3 in range(0, num1 + 1):
            value_of_maximum = max(arr)
            index_of_maximum = np.argmax(arr)
            matrix1.append(index_of_maximum)
            arr[index_of_maximum] = 0
        if ((num2 + 1) % 2) == 0:
            check = ((num2 + 1) / 2) - 1
        else:
            check = int((num2 + 1) / 2)
        t = check in matrix1
        if True == t:
            count = count + 1
    print('accuracy:', (count / xx) * 100)
    acc = (count / xx) * 100
    accuracy1.append(acc)
print('Rank based accuracy from 1 to 20: ', accuracy1)
with open('/content/drive/My Drive/GEINet_and_PEINet/GEI_Net_CSV/rank_based_GEINet_casia_c_fq2nm.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(accuracy1)