import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv
import PIL
from PIL import Image
import os
from keras.layers import Input, Convolution2D, MaxPooling2D, Flatten, Conv2D, InputLayer,MaxPool2D,Dropout,Dense
from keras import optimizers
from keras.initializers import Constant
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from numpy.random import seed
from skimage.data import imread
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score
from keras.layers import Dropout
seed(1)

a=64
b=64

'function for image to csv conversion' 

def imagetocsv(fpath,name):
    def createFileList(myDir, format='.jpg'):
        fileList = []
        print(myDir)
        for root, dirs, files in os.walk(myDir, topdown=False):
            for name in files:
                if name.endswith(format):
                    fullName = os.path.join(root, name)
                    fileList.append(fullName)
        return fileList
    
    fileList = createFileList(fpath)
    
    basewidth = a
    hsize =b
    
    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    
    for file in fileList:
        print(file)
        img = Image.open(file)
        #img = Image.open('C:\\Users\\1018090\\Downloads\\Convolutional-Neural-Networks\\Convolutional_Neural_Networks\\dataset\\test_set\\cats\\cat.4001.jpg')
        #wpercent = (basewidth / float(img.size[0]))
        #hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
        img.save(fpath+'\\resized_image.jpg')
        img = mpimg.imread(fpath+'\\resized_image.jpg')     
        gray = rgb2gray(img)
        value = gray.reshape(basewidth*hsize,1)
        value = value.flatten()
        with open(fpath+'\\'+name+'.csv', 'a',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(value)
    df = pd.read_csv(fpath+'\\'+name+'.csv')
    df.to_csv(fpath+'\\'+name+'.csv', index=False)      


'image to csv conversion' 

imagetocsv('C:\\Users\\1018090\\Downloads\\Convolutional-Neural-Networks\\Convolutional_Neural_Networks\\dataset\\test_set\\cats','test_cat')
imagetocsv('C:\\Users\\1018090\\Downloads\\Convolutional-Neural-Networks\\Convolutional_Neural_Networks\\dataset\\test_set\\dogs','test_dog')

imagetocsv('C:\\Users\\1018090\\Downloads\\Convolutional-Neural-Networks\\Convolutional_Neural_Networks\\dataset\\training_set\\cats','train_cat')
imagetocsv('C:\\Users\\1018090\\Downloads\\Convolutional-Neural-Networks\\Convolutional_Neural_Networks\\dataset\\training_set\\dogs','train_dog')

Test_Cat=pd.read_csv('C:\\Users\\1018090\\Downloads\\Convolutional-Neural-Networks\\Convolutional_Neural_Networks\\dataset\\test_set\\cats\\test_cat.csv')
Test_Dog=pd.read_csv('C:\\Users\\1018090\\Downloads\\Convolutional-Neural-Networks\\Convolutional_Neural_Networks\\dataset\\test_set\\dogs\\test_dog.csv')
Train_Cat=pd.read_csv('C:\\Users\\1018090\\Downloads\\Convolutional-Neural-Networks\\Convolutional_Neural_Networks\\dataset\\training_set\\cats\\train_cat.csv')
Train_Dog=pd.read_csv('C:\\Users\\1018090\\Downloads\\Convolutional-Neural-Networks\\Convolutional_Neural_Networks\\dataset\\training_set\\dogs\\train_dog.csv')

'funcn to label target and index'

def lab_ind(df,label):
    df['Label']=label
    #list=[]
    #for x in range(len(Test_Cat.index)):
    #    list.append(x)        
    list=[]
    for x in range(len(df.columns)-1):
        list.append(x)
    
    list.append('Label')
    df.columns=list
    #Test_Cat=Test_Cat.reindex(columns)

'label and index'

lab_ind(Test_Cat,0)
lab_ind(Test_Dog,1)
lab_ind(Train_Cat,0)
lab_ind(Train_Dog,1)


    
'dataset labels and attributes'

train_images_full=Train_Cat.append(Train_Dog, ignore_index=True)
test_images_full=Test_Cat.append(Test_Dog, ignore_index=True)

train_images=train_images_full.iloc[:,:-1]
test_images=test_images_full.iloc[:,:-1]

train_labels=train_images_full.iloc[:,-1]
test_labels=test_images_full.iloc[:,-1]

class_names = ['Cats', 'dogs']

train_images = train_images / 255.0
test_images = test_images / 255.0

'Ploting to check images'

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    img=train_images.iloc[i].as_matrix()
    img=img.reshape((64,64))
    plt.imshow(img,cmap='gray')
    plt.title("cat")
    
    
train_labels=np_utils.to_categorical(train_labels,2)
test_labels=np_utils.to_categorical(test_labels,2)  

'Build Model'
 
clf = Sequential()
clf.add(InputLayer(input_shape=(1,a, b)))
clf.add(BatchNormalization())
clf.add(Conv2D(32, (2, 2), padding='same', bias_initializer=Constant(0.01), kernel_initializer='random_uniform'))
clf.add(MaxPool2D(padding='same'))
clf.add(BatchNormalization())
clf.add(Conv2D(64, (2, 2), padding='same', bias_initializer=Constant(0.01), kernel_initializer='random_uniform'))
clf.add(MaxPool2D(padding='same'))
clf.add(BatchNormalization())
clf.add(Conv2D(128, (2, 2), padding='same', bias_initializer=Constant(0.01), kernel_initializer='random_uniform'))
clf.add(MaxPool2D(padding='same'))
clf.add(Dropout(0.5, input_shape=(60,)))
clf.add(Flatten())
clf.add(Dense(1024,activation='relu',bias_initializer=Constant(0.01), kernel_initializer='random_uniform', ))
clf.add(Dense(2, activation='softmax'))
clf.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

clf.fit(train_images.values.flatten().reshape(7998,1,a,b), train_labels, epochs=50)


test=test_images.values.flatten().reshape(1998,1,a,b)
train=train_images.values.flatten().reshape(7998,1,a,b)
train_loss, train_acc = clf.evaluate(train, train_labels)

print('Train accuracy:', train_acc)


'prediction'

predictions = clf.predict(test)
predicted_label = np.argmax(predictions[i])

predicted_label_all=[]
for i in range(len(predictions)):
    x=np.argmax(predictions[i])
    predicted_label_all.append(x)

test_labelss=test_images_full.iloc[:,-1]
accuracy_score(predicted_label_all,test_labelss)


 


        







