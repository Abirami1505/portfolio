import numpy as np
from flask import Flask,request,render_template
import pickle
app=Flask(__name__)
model=pickle.load(open("C:/Users/abira/OneDrive/Desktop/heart disease ml/model.pkl",'rb'))
@app.route('/')
def home():
  return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
  feature=[]
  for i in request.form.values():
      feature.append(float(i))
  feature=[np.array(feature)]
  prediction=model.predict(feature)
  if prediction[0]==1:
    output="positive"
  else:
    output="negative"
  return render_template('index.html',prediction_text='heart disease : {}'.format(output))
if __name__=="__main__":
  app.run()

