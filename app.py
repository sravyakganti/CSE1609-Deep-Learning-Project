# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 22:42:38 2020

@author: aswitha eddy
"""


import numpy as np
import os
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
global graph
graph = tf.get_default_graph()
from flask import Flask , request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
model = load_model("skindisease.h5")

@app.route('/')
def index():
    return render_template('base3.html')

@app.route('/prediction')
def y():
    return render_template('y.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        
        with graph.as_default():
            preds = model.predict_classes(x)
            4
            
            print("prediction",preds)
            
        index = ['acne','blister','chikenpox','melanoma','vitiligo',]
        
        text = "the predicted disease is : " + str(index[preds[0]])
        
    return text
if __name__ == '__main__':
    app.run(debug = True, threaded = False)