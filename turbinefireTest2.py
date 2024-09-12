from keras.models import load_model
model=load_model("C:/Users/abira/OneDrive/Desktop/turbinefire2.h5")

import numpy as np
import matplotlib.pyplot as plt
import os  
input_path=[]
label=[]

for class_name in os.listdir("C:/Users/abira/OneDrive/Desktop/turbinefire/test"):
    for path in os.listdir("C:/Users/abira/OneDrive/Desktop/turbinefire/test/"+class_name):
        if class_name=='not_fire':
            label.append(0)
        else:
            label.append(1)
        input_path.append(os.path.join("C:/Users/abira/OneDrive/Desktop/turbinefire/test", class_name,path))
input_path=np.array(input_path)
label=np.array(label)

result=[]
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
for i in input_path:
    image=load_img(i,target_size=(128,128))
    image=img_to_array(image)
    image=image.reshape(1, 128, 128, 3)
    r=model.predict(image)
    result.append(int(r[0][0]))
result=np.array(result) 

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix
acc = accuracy_score(label, result)
print("The accuracy is ",acc)

prec = precision_score(label, result)
print("The precision is ",prec)

rec = recall_score(label, result)
print("The recall is ",rec)

f1 = f1_score(label, result)
print("The F1-Score is ",f1)

MCC = matthews_corrcoef(label, result)
print("The Matthews correlation coefficient is ",MCC) 

import seaborn as sns
LABELS = ['fire', 'not_fire']
conf_matrix = confusion_matrix(label, result)
plt.figure(figsize =(10, 10))
sns.heatmap(conf_matrix, xticklabels = LABELS,
            yticklabels = LABELS, annot = True, fmt ="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()



