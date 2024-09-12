from flask import Flask, render_template, request, flash, redirect
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
import tensorflow_hub as hub
import numpy as np
import cv2

model = load_model("C:/Users/abira/OneDrive/Desktop/datasets/epoch_25_kfold.h5")

app= Flask(__name__)
app.config['UPLOAD_FOLDER']= 'static/'
app.secret_key="SASTRA"

ALLOWED_EXTENSIONS = set(['png','jpg','jpeg','gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/uploader", methods=['POST'])
def uploader():
    #if request.method=='POST':
    f=request.files['file']
    if allowed_file(f):
        f.filename = "image.jpg"
    else:
        flash("file type not acceptable")
        return redirect(request.url)
    f.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename)))
    input_image= cv2.imread("static/image.jpg")
    input_image_resize= cv2.resize(input_image, (128,128))
    input_image_scaled=input_image_resize/255
    image_reshaped = np.reshape(input_image_scaled,[1,128,128,3])
    input_prediction=model.predict(image_reshaped)
    if input_prediction[0]<0.5:
        result= 'CAT'
    elif input_prediction[0]>0.5:
        result = 'DOG'
    else:
        result="none"
    pic1= os.path.join(app.config['UPLOAD_FOLDER'], 'image.jpg')
    return render_template("uploaded.html", result=result,input_image=pic1)

if __name__ == '__main__':
    app.run(debug=True)

