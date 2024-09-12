import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os    
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import random
import warnings
warnings.filterwarnings('ignore')
from keras_preprocessing.image import ImageDataGenerator
import sklearn.model_selection as sklearn

#getting images
input_path=[]
label=[]

for class_name in os.listdir("C:/Users/abira/OneDrive/Desktop/datasets/training_set"):
    for path in os.listdir("C:/Users/abira/OneDrive/Desktop/datasets/training_set/"+class_name):
        if class_name=='cats':
            label.append(0)
        else:
            label.append(1)
        input_path.append(os.path.join("C:/Users/abira/OneDrive/Desktop/datasets/training_set", class_name,path))
input_path=np.array(input_path)
label=np.array(label)

'''input_path_test=[]
label_test=[]

for class_name in os.listdir("C:/Users/abira/OneDrive/Desktop/datasets/testSet"):
    for path in os.listdir("C:/Users/abira/OneDrive/Desktop/datasets/testSet/"+class_name):
        if class_name=='cats':
            label_test.append(0)
        else:
            label_test.append(1)
        input_path_test.append(os.path.join("C:/Users/abira/OneDrive/Desktop/datasets/testSet", class_name,path))

'''
#train and test generator    
train_generator=ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
test_generator=ImageDataGenerator(rescale=1./255)


from keras import Sequential
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense

#defining model
model= Sequential([
    Conv2D(16,(3, 3), activation='relu',input_shape=(128,128,3)),
    MaxPool2D((2,2)), #Dropout(0.2),
    Conv2D(32,(3, 3), activation='relu'),
    MaxPool2D((2,2)), #Dropout(0.2),
    Conv2D(64,(3, 3), activation='relu'),
    MaxPool2D((2,2)), #Dropout(0.2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')])

#compiling
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
print(model.summary())

#accuracy score
scores=[]

#defining kfold
kfold=sklearn.KFold(5,shuffle=True,random_state=1)

#kfold iteration
for train_index,test_index in kfold.split(input_path):
    x_train,x_test=input_path[train_index],input_path[test_index]
    y_train,y_test=label[train_index],label[test_index]
    #train dataframe
    df=pd.DataFrame()
    df['images']=x_train
    df['label']=y_train
    df=df.sample(frac=1).reset_index(drop=True)
    df['label']=df['label'].astype('str')
    print("train:",df.head())
    #test dataframe
    df_test=pd.DataFrame()
    df_test['images']=x_test
    df_test['label']=y_test
    df_test=df_test.sample(frac=1).reset_index(drop=True)
    df_test['label']=df_test['label'].astype('str')
    print("test:",df_test.head())
    
    #iterators
    train_iterator=train_generator.flow_from_dataframe(df,x_col='images',y_col='label',target_size=(128,128),batch_size=64,class_mode='binary')
    test_iterator=test_generator.flow_from_dataframe(df_test,x_col='images',y_col='label',target_size=(128,128),batch_size=64,class_mode='binary')
    
    #fitting the model
    history = model.fit(train_iterator,epochs=1,validation_data=test_iterator)
    
    #accuracy score
    scores.append({'acc':np.average(history.history['accuracy']),'val_acc':np.average(history.history['val_accuracy'])})

#saving model
model.save("C:/Users/abira/OneDrive/Desktop/datasets/final_model1.h5")

#accuracy visualization
train=[]
validation=[]
plt.subplot(1,1,1)
for s in scores:
    train.append(s['acc'])
    validation.append(s['val_acc'])
plt.plot(train, color='blue', label='train')
plt.plot(validation, color='red', label='validation')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validation'])
plt.show()

from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras.models import load_model
#prediction
img=load_img("C:/Users/abira/OneDrive/Desktop/datasets/training_set/dogs/dog.141.jpg",target_size=(128,128))
img=img_to_array(img)
img=img.reshape(1, 128, 128, 3)
model=load_model("C:/Users/abira/OneDrive/Desktop/datasets/final_model.h5")
result=model.predict(img)
print(result[0])